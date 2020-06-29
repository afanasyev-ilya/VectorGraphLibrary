/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../graph_library.h"
#include <iostream>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void run(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, GraphPrimitivesNEC &graph_API,
        FrontierNEC &frontier, _TEdgeWeight *distances)
{
    double t1, t2;
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    float *collective_outgoing_weights = graph_API.get_collective_weights(_graph, frontier);

    auto edge_op_push = [outgoing_weights, distances](int src_id, int dst_id, int local_edge_pos,
                                                      long long int global_edge_pos, int vector_index, DelayedWriteNEC &delayed_write)
    {
        float weight = outgoing_weights[global_edge_pos];
        float dst_weight = distances[dst_id];
        float src_weight = distances[src_id];
        if(dst_weight > src_weight + weight)
        {
            distances[dst_id] = src_weight + weight;
        }
    };
    auto edge_op_collective_push = [collective_outgoing_weights, distances, vertices_count]
            (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos,
             int vector_index, DelayedWriteNEC &delayed_write)
    {
        float weight = collective_outgoing_weights[global_edge_pos];
        float dst_weight = distances[dst_id];
        float src_weight = distances[src_id];
        if(dst_weight > src_weight + weight)
        {
            distances[dst_id] = src_weight + weight;
        }
    };

    t1 = omp_get_wtime();
    graph_API.advance(_graph, frontier, edge_op_push, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
                      edge_op_collective_push, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
    t2 = omp_get_wtime();
    cout << "time: " << 1000.0*(t2 - t1) << " ms" << endl << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void test_gather(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    _TEdgeWeight *distances;
    MemoryAPI::allocate_array(&distances, vertices_count);

    GraphPrimitivesNEC graph_API;
    FrontierNEC frontier(vertices_count);

    frontier.set_all_active();

    auto init_distances = [distances] (int src_id, int connections_count, int vector_index)
    {
        distances[src_id] = src_id;
    };
    graph_API.compute(_graph, frontier, init_distances);

    cout << "------------------ FIRST advance -----------------------------" << endl;
    run(_graph, graph_API, frontier, distances);

    cout << "------------------ SECOND advance -----------------------------" << endl;
    run(_graph, graph_API, frontier, distances);

    cout << "------------------ odd-even advance -----------------------------" << endl;
    auto odd_even = [] (int src_id)->int
    {
        int result = NOT_IN_FRONTIER_FLAG;
        if(src_id % 2 == 0)
            result = IN_FRONTIER_FLAG;
        return result;
    };
    graph_API.generate_new_frontier(_graph, frontier, odd_even);
    run(_graph, graph_API, frontier, distances);

    cout << "------------------ %4 advance -----------------------------" << endl;
    auto per_four = [] (int src_id)->int
    {
        int result = NOT_IN_FRONTIER_FLAG;
        if(src_id % 4 == 0)
            result = IN_FRONTIER_FLAG;
        return result;
    };
    graph_API.generate_new_frontier(_graph, frontier, per_four);
    run(_graph, graph_API, frontier, distances);

    cout << "------------------ %8 advance -----------------------------" << endl;
    auto per_eight = [] (int src_id)->int
    {
        int result = NOT_IN_FRONTIER_FLAG;
        if(src_id % 8 == 0)
            result = IN_FRONTIER_FLAG;
        return result;
    };
    graph_API.generate_new_frontier(_graph, frontier, per_eight);
    run(_graph, graph_API, frontier, distances);

    MemoryAPI::free_array(distances);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        cout << "Gather test..." << endl;

        // parse args
        AlgorithmCommandOptionsParser parser;
        parser.parse_args(argc, argv);

        ExtendedCSRGraph<int, float> graph;
        if(parser.get_compute_mode() == GENERATE_NEW_GRAPH)
        {
            EdgesListGraph<int, float> rand_graph;
            int vertices_count = pow(2.0, parser.get_scale());
            long long edges_count = vertices_count * parser.get_avg_degree();
            GraphGenerationAPI<int, float>::R_MAT(rand_graph, vertices_count, edges_count, 57, 19, 19, 5, DIRECTED_GRAPH);
            graph.import_graph(rand_graph, VERTICES_SORTED, EDGES_SORTED, VECTOR_LENGTH, PULL_TRAVERSAL);
        }
        else if(parser.get_compute_mode() == LOAD_GRAPH_FROM_FILE)
        {
            double t1 = omp_get_wtime();
            if(!graph.load_from_binary_file(parser.get_graph_file_name()))
                throw "ERROR: graph file not found";
            double t2 = omp_get_wtime();
            cout << "file " << parser.get_graph_file_name() << " loaded in " << t2 - t1 << " sec" << endl;
        }

        test_gather(graph);
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
