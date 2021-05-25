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
void mpi_sssp(VectCSRGraph &_graph, EdgesArray_Vect<_T> &_weights,
              VerticesArray<_T> &_distances, int _source_vertex)
{
    #ifdef __USE_MPI__
    vgl_library_data.allocate_exchange_buffers(_distances.size(), sizeof(_T));
    vgl_library_data.set_data_exchange_policy(RECENTLY_CHANGED);
    #endif

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

    graph_API.compute(_graph, frontier, init_distances);

    int changes = 0, iterations_count = 0;
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

        auto min_op = [](float _a, float _b)->float
        {
            return vect_min(_a, _b);
        };
        auto max_op = [](int _a, int _b)->int
        {
            return vect_max(_a, _b);
        };

        vgl_library_data.exchange_data(_distances.get_ptr(), _graph.get_vertices_count(), min_op,  prev_distances.get_ptr());
        vgl_library_data.exchange_data(&changes, 1, max_op);
    }
    while(changes);
    MPI_Barrier(MPI_COMM_WORLD);
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

        int source_vertex = graph.select_random_vertex(ORIGINAL);//TODO BCAST

        performance_stats.reset_timers();
        mpi_sssp(graph, weights, distances, source_vertex);
        performance_stats.update_timer_stats();
        performance_stats.print_timers_stats();

        performance_stats.reset_timers();
        SSSP::nec_dijkstra_all_active_pull(graph, weights, distances, source_vertex);
        performance_stats.update_timer_stats();
        performance_stats.print_timers_stats();

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
