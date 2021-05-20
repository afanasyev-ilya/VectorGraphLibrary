/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0
#define VECTOR_ENGINE_THRESHOLD_VALUE VECTOR_LENGTH*MAX_SX_AURORA_THREADS*128

#ifdef __USE_NEC_SX_AURORA__
#define VECTOR_CORE_THRESHOLD_VALUE 3*VECTOR_LENGTH
#endif

#ifdef __USE_MULTICORE__
#define VECTOR_CORE_THRESHOLD_VALUE 5*VECTOR_LENGTH
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
int estimate_changes_count(_T *_new, _T *_old, int _size)
{
    int changes_count = 0;
    #pragma omp parallel for reduction(+: changes_count)
    for(int i = 0; i < _size; i++)
    {
        if(_new[i] != _old[i])
            changes_count++;
    }
    return changes_count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int get_recv_size(int _send_size, int _source, int _dest)
{
    int recv_size = 0;
    MPI_Sendrecv(&_send_size, 1, MPI_INT,
                 _dest, 0, &recv_size, 1, MPI_INT,
                 _source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return recv_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double t_small = 0;
double t_large = 0;

template <typename _T, typename UpdateOp>
void exchange_data(_T *_data, _T *_old, _T *_buffer, int _size, UpdateOp &&_update_op)
{
    int source = (vgl_library_data.get_mpi_rank() + 1);
    int dest = (vgl_library_data.get_mpi_rank() - 1);
    if(source >= vgl_library_data.get_mpi_proc_num())
        source = 0;
    if(dest < 0)
        dest = vgl_library_data.get_mpi_proc_num() - 1;

    int send_size = estimate_changes_count(_data, _old, _size);
    int recv_size = get_recv_size(send_size, source, dest);
    //cout << "send_size: " << ((double)send_size) / _size << endl;
    //cout << "recv_size: " << ((double)recv_size) / _size << endl;
    //cout << "Size: " << _size << endl;

    double t1 = omp_get_wtime();
    MPI_Sendrecv(_data, send_size, MPI_FLOAT,
                 dest, 0, _buffer, recv_size, MPI_FLOAT,
                 source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    double t2 = omp_get_wtime();
    cout << "SMALL send time: " << (t2 - t1)*1000 << " ms" << endl;
    cout << sizeof(float)*(recv_size + send_size)/((t2 - t1)*1e9) << " GB/s" << endl;
    t_small += t2 - t1;

    t1 = omp_get_wtime();
    MPI_Sendrecv(_data, _size, MPI_FLOAT,
                 dest, 0, _buffer, _size, MPI_FLOAT,
                 source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    t2 = omp_get_wtime();
    cout << "LARGE send time: " << (t2 - t1)*1000 << " ms" << endl;
    cout << sizeof(float)*(_size + _size)/((t2 - t1)*1e9) << " GB/s" << endl;
    t_large += t2 - t1;

    #pragma _NEC cncall
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma _NEC gather_reorder
    #pragma omp parallel for
    for(int i = 0; i < _size; i++)
    {
        _data[i] = _update_op(_buffer[i], _data[i]);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void mpi_sssp(VectCSRGraph &_graph, EdgesArray_Vect<_T> &_weights,
              VerticesArray<_T> &_distances, int _source_vertex)
{
    VGL_GRAPH_ABSTRACTIONS graph_API(_graph);
    VGL_FRONTIER frontier(_graph);

    VerticesArray<_T> prev_distances(_graph);

    _source_vertex = _graph.reorder(_source_vertex, ORIGINAL, SCATTER);
    graph_API.change_traversal_direction(SCATTER, _distances, frontier);

    Timer tm;
    tm.start();

    _T inf_val = std::numeric_limits<_T>::max() - MAX_WEIGHT;
    auto init_distances = [&_distances, _source_vertex, inf_val] __VGL_COMPUTE_ARGS__
    {
        if(src_id == _source_vertex)
            _distances[src_id] = 0;
        else
            _distances[src_id] = inf_val;
    };
    frontier.set_all_active();

    //
    int mpi_rank = vgl_library_data.get_mpi_rank();
    res = _graph.get_mpi_thresholds(mpi_rank, SCATTER);
    ve_mpi_borders = frontier.get_vector_engine_mpi_thresholds();
    vc_mpi_borders = frontier.get_vector_core_mpi_thresholds();
    coll_mpi_borders = frontier.get_collective_mpi_thresholds();
    //

    MPI_Barrier(MPI_COMM_WORLD);

    graph_API.compute(_graph, frontier, init_distances);

    double compute_time = 0;
    int changes = 0, iterations_count = 0;
    double tt1 = omp_get_wtime();
    do
    {
        changes = 0;
        iterations_count++;

        auto save_old_distances = [&_distances, &prev_distances] __VGL_COMPUTE_ARGS__
        {
            prev_distances[src_id] = _distances[src_id];
        };
        graph_API.compute(_graph, frontier, save_old_distances);

        auto edge_op_push = [&_distances, &_weights] __VGL_SCATTER_ARGS__
        {
            _T weight = _weights[global_edge_pos];
            _T src_weight = _distances[src_id];
            if(_distances[dst_id] > src_weight + weight)
            {
                _distances[dst_id] = src_weight + weight;
            }
        };

        double t1_loc = omp_get_wtime();
        graph_API.scatter(_graph, frontier, edge_op_push);

        auto reduce_changes = [&_distances, &prev_distances]__VGL_REDUCE_INT_ARGS__
        {
            int result = 0.0;
            if(prev_distances[src_id] != _distances[src_id])
            {
                result = 1;
            }
            return result;
        };
        changes = graph_API.reduce<int>(_graph, frontier, reduce_changes, REDUCE_SUM);
        double t2_loc = omp_get_wtime();
        cout << "compute time: " << (t2_loc - t1_loc)*1000 << " ms" << endl;
        compute_time += t2_loc - t1_loc;

        cout << 100.0 * (double)changes / _graph.get_vertices_count() << " % SAVING" << endl;

        int changes_buffer = 0;
        auto min_op = [](float _a, float _b)->float
        {
            return vect_min(_a, _b);
        };
        auto max_op = [](int _a, int _b)->int
        {
            return vect_max(_a, _b);
        };
        exchange_data(_distances.get_ptr(), prev_distances.get_ptr(), prev_distances.get_ptr(), _graph.get_vertices_count(), min_op);
        exchange_data(&changes, &changes_buffer, &changes_buffer, 1, max_op);
    }
    while(changes);
    double tt2 = omp_get_wtime();

    cout << "compute_time: " << compute_time * 1000 << " ms" << endl;
    cout << "t_small: " << t_small * 1000 << " ms" << endl;
    cout << "t_large: " << t_large * 1000 << " ms" << endl;

    //performance_stats.update_timer_stats();
    //performance_stats.print_timers_stats();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        vgl_library_data.init(argc, argv);

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        VectCSRGraph graph;
        if(parser.get_compute_mode() == GENERATE_NEW_GRAPH)
        {
            EdgesListGraph el_graph;
            int v = pow(2.0, parser.get_scale());
            if(parser.get_graph_type() == RMAT)
                GraphGenerationAPI::R_MAT(el_graph, v, v * parser.get_avg_degree(), 57, 19, 19, 5, DIRECTED_GRAPH);
            else if(parser.get_graph_type() == RANDOM_UNIFORM)
                GraphGenerationAPI::random_uniform(el_graph, v, v * parser.get_avg_degree(), DIRECTED_GRAPH);
            graph.import(el_graph);
        }
        else if(parser.get_compute_mode() == LOAD_GRAPH_FROM_FILE)
        {
            Timer tm;
            tm.start();
            if(!graph.load_from_binary_file(parser.get_graph_file_name()))
                throw "Error: graph file not found";
            tm.end();
            tm.print_time_stats("Graph load");
        }

        EdgesArray_Vect<float> weights(graph);
        weights.set_all_constant(1.0);

        VerticesArray<float> distances(graph);

        performance_stats.reset_timers();

        int source_vertex = graph.select_random_vertex(ORIGINAL);//TODO BCAST

        mpi_sssp(graph, weights, distances, source_vertex);

        if(parser.get_check_flag())
        {
            VerticesArray<float> check_distances(graph, SCATTER);
            ShortestPaths::seq_dijkstra(graph, weights, check_distances, source_vertex);
            verify_results(distances, check_distances);
        }

        vgl_library_data.finalize();
    }
    catch (string error)
    {
        cout << error << endl;
    }
    catch (const char * error)
    {
        cout << error << endl;
    }
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
