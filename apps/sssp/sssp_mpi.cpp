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

/*inline int copy_if_changed(const int *_in_data,
                           int *_out_data,
                           int *_tmp_buffer,
                           const int _buffer_size,
                           const int _start,
                           const int _end,
                           const int _desired_value,
                           const int _threads_count = MAX_SX_AURORA_THREADS)
{
    int size = _end - _start;
    int elements_per_thread = (_buffer_size - 1)/_threads_count + 1;
    int elements_per_vector = (elements_per_thread - 1)/VECTOR_LENGTH + 1;
    int shifts_array[MAX_SX_AURORA_THREADS];

    int elements_count = 0;
    #pragma omp parallel num_threads(_threads_count) shared(elements_count)
    {
        int tid = omp_get_thread_num();
        int start_pointers_reg[VECTOR_LENGTH];
        int current_pointers_reg[VECTOR_LENGTH];
        int last_pointers_reg[VECTOR_LENGTH];

        #pragma _NEC vreg(start_pointers_reg)
        #pragma _NEC vreg(current_pointers_reg)
        #pragma _NEC vreg(last_pointers_reg)

        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            start_pointers_reg[i] = tid * elements_per_thread + i * elements_per_vector;
            current_pointers_reg[i] = tid * elements_per_thread + i * elements_per_vector;
            last_pointers_reg[i] = tid * elements_per_thread + i * elements_per_vector;
        }

        #pragma omp for schedule(static)
        for(int vec_start = _start; vec_start < _end; vec_start += VECTOR_LENGTH)
        {
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = vec_start + i;
                if((src_id < _end) && (_in_data[src_id] == _desired_value))
                {
                    _tmp_buffer[current_pointers_reg[i]] = src_id;
                    current_pointers_reg[i]++;
                }
            }
        }

        int max_difference = 0;
        int save_values_per_thread = 0;
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            int difference = current_pointers_reg[i] - start_pointers_reg[i];
            save_values_per_thread += difference;
            if(difference > max_difference)
                max_difference = difference;
        }

        shifts_array[tid] = save_values_per_thread;
        #pragma omp barrier

        #pragma omp master
        {
            int cur_shift = 0;
            for(int i = 1; i < _threads_count; i++)
            {
                shifts_array[i] += shifts_array[i - 1];
            }

            elements_count = shifts_array[_threads_count - 1];

            for(int i = (_threads_count - 1); i >= 1; i--)
            {
                shifts_array[i] = shifts_array[i - 1];
            }
            shifts_array[0] = 0;
        }

        #pragma omp barrier

        int tid_shift = shifts_array[tid];
        int *private_ptr = &(_out_data[tid_shift]);

        int local_pos = 0;
        #pragma _NEC novector
        for(int pos = 0; pos < max_difference; pos++)
        {
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int loc_size = current_pointers_reg[i] - start_pointers_reg[i];

                if(pos < loc_size)
                {
                    private_ptr[local_pos] = _tmp_buffer[last_pointers_reg[i]];
                    last_pointers_reg[i]++;
                    local_pos++;
                }
            }
        }
    }

    return elements_count;
}*/

template <typename _T>
int prepare_exchange_data(_T *_new, _T *_old, char *_buffer, int _size)
{
    int changes_count = 0;
    #pragma omp parallel for reduction(+: changes_count)
    for(int i = 0; i < _size; i++)
    {
        if(_new[i] != _old[i])
        {
            changes_count++;
        }
    }

    _T *data_buffer = (_T*) _buffer;
    int *index_buffer = (int*)(&_buffer[changes_count*sizeof(_T)]);
    int write_ptr = 0;
    for(int i = 0; i < _size; i++)
    {
        if(_new[i] != _old[i])
        {
            data_buffer[write_ptr] = _new[i];
            index_buffer[write_ptr] = i;
            write_ptr++;
        }
    }

    return changes_count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void parse_received_data(_T *_data, char *_buffer, int _recv_size)
{
    _T *data_buffer = (_T*) _buffer;
    int *index_buffer = (int*)(&_buffer[_recv_size*sizeof(_T)]);

    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma omp parallel for
    for(int i = 0; i < _recv_size; i++)
    {
        _data[index_buffer[i]] = data_buffer[i];
    }
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

    char *send_buffer = new char[_size*sizeof(_T) + _size*sizeof(int)];
    char *recv_buffer = new char[_size*sizeof(_T) + _size*sizeof(int)];

    int send_elements = prepare_exchange_data(_data, _old, send_buffer, _size);
    int recv_elements = get_recv_size(send_elements, source, dest);
    size_t send_size = (sizeof(_T) + sizeof(int))*send_elements;
    size_t recv_size = (sizeof(_T) + sizeof(int))*recv_elements;


    double t1 = omp_get_wtime();
    MPI_Sendrecv(send_buffer, send_size, MPI_CHAR,
                 dest, 0, recv_buffer, recv_size, MPI_CHAR,
                 source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    double t2 = omp_get_wtime();
    cout << "SMALL send time: " << (t2 - t1)*1000 << " ms" << endl;
    cout << (recv_size + send_size)/((t2 - t1)*1e9) << " GB/s" << endl;
    t_small += t2 - t1;

    parse_received_data(_data, recv_buffer, recv_elements);

    delete []send_buffer;
    delete []recv_buffer;
r
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

template <typename _T, typename UpdateOp>
void full_exchange_data(_T *_data, _T *_old, _T *_buffer, int _size, UpdateOp &&_update_op)
{
    int source = (vgl_library_data.get_mpi_rank() + 1);
    int dest = (vgl_library_data.get_mpi_rank() - 1);
    if(source >= vgl_library_data.get_mpi_proc_num())
        source = 0;
    if(dest < 0)
        dest = vgl_library_data.get_mpi_proc_num() - 1;

    double t1 = omp_get_wtime();
    MPI_Sendrecv(_data, _size, MPI_FLOAT,
                 dest, 0, _buffer, _size, MPI_FLOAT,
                 source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    double t2 = omp_get_wtime();
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
