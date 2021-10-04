/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0
#define VECTOR_ENGINE_THRESHOLD_VALUE 2147483646
#define VECTOR_CORE_THRESHOLD_VALUE 5*VECTOR_LENGTH

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void test_on_full_graph(Parser &parser)
{
    // prepare graph
    VGL_Graph graph(VGL_RUNTIME::select_graph_format(parser), VGL_RUNTIME::select_graph_optimizations(parser));

    EdgesContainer edges_container;
    if(!edges_container.load_from_binary_file(parser.get_graph_file_name()))
        throw "Error: edges container file not found";

    graph.import(edges_container);

    cout << "MPI rank " << edges_container.get_edges_count()/1e6 << "m edges" << endl;

    VGL_GRAPH_ABSTRACTIONS graph_API(graph, SCATTER);
    VGL_FRONTIER frontier(graph, SCATTER);
    VerticesArray<float> distances(graph, SCATTER);

    auto init_distances = [distances] __VGL_COMPUTE_ARGS__ {
        distances[src_id] = src_id;
    };
    frontier.set_all_active();
    graph_API.compute(graph, frontier, init_distances);

    auto edge_op_push = [distances] __VGL_SCATTER_ARGS__ {
        float weight = 1;
        float src_weight = distances[src_id];
        if(distances[dst_id] > src_weight + weight)
        {
            distances[dst_id] = src_weight + weight;
        }
    };

    double t1 = omp_get_wtime();
    graph_API.scatter(graph, frontier, edge_op_push);
    double t2 = omp_get_wtime();

    cout << "advance_time full : " << (t2 - t1) * 1000.0 << " ms" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void test_on_partitioned_graph(Parser &parser)
{
    // prepare graph
    VGL_Graph graph(VGL_RUNTIME::select_graph_format(parser), VGL_RUNTIME::select_graph_optimizations(parser));

    EdgesContainer edges_container;
    if(!edges_container.load_from_binary_file(parser.get_graph_file_name()))
        throw "Error: edges container file not found";

    long long old_edges_count = edges_container.get_edges_count();

    #ifdef __USE_MPI__
    MPI_partitioner partitioner(vgl_library_data.get_mpi_proc_num(), parser.get_partitioning_mode());
    partitioner.run(edges_container);
    cout << "MPI rank " << vgl_library_data.get_mpi_rank() << " part size: " <<
                100.0*((double)edges_container.get_edges_count()/old_edges_count) << " %" << endl;
    cout << "MPI rank " << edges_container.get_edges_count()/1e6 << "m edges" << endl;
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    graph.import(edges_container);

    VGL_GRAPH_ABSTRACTIONS graph_API(graph, SCATTER);
    VGL_FRONTIER frontier(graph, SCATTER);
    VerticesArray<float> distances(graph, SCATTER);

    auto init_distances = [distances] __VGL_COMPUTE_ARGS__ {
        distances[src_id] = src_id;
    };
    frontier.set_all_active();
    graph_API.compute(graph, frontier, init_distances);

    auto edge_op_push = [distances] __VGL_SCATTER_ARGS__ {
        float weight = 1;
        float src_weight = distances[src_id];
        if(distances[dst_id] > src_weight + weight)
        {
            distances[dst_id] = src_weight + weight;
        }
    };

    double t1 = omp_get_wtime();
    graph_API.scatter(graph, frontier, edge_op_push);
    double t2 = omp_get_wtime();

    cout << "advance_time partitioned : " << (t2 - t1) * 1000.0 << " ms" << endl;
}

int main(int argc, char **argv)
{
    try
    {
        VGL_RUNTIME::init_library(argc, argv);
        VGL_RUNTIME::info_message("MPI TEST");

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        test_on_full_graph(parser);
        MPI_Barrier(MPI_COMM_WORLD);
        test_on_partitioned_graph(parser);

        // переписать frontier - очень неудобно
        // сделать свои функции - быстро

        VGL_RUNTIME::finalize_library();
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
