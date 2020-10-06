/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 6.0

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//#define __PRINT_API_PERFORMANCE_STATS__
#define __PRINT_SAMPLES_PERFORMANCE_STATS__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../graph_library.h"
#include <iostream>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void print_first_edges(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_rand_graph)
{
    int *src_ids = _rand_graph.get_src_ids();
    int *dst_ids = _rand_graph.get_dst_ids();

    int len = min((int)15, (int)_rand_graph.get_edges_count());
    for(int i = 0; i < len; i++)
    {
        cout << src_ids[i] << " " << dst_ids[i] << endl;
    }
    cout << endl;
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

        EdgesListGraph<int, float> graph;
        int v = pow(2.0, parser.get_scale());
        GraphGenerationAPI<int, float>::random_uniform(graph, v, v * parser.get_avg_degree(), DIRECTED_GRAPH);
        cout << "graph generated!" << endl;

        VectCSRGraph<int, float> vect_csr_graph;
        vect_csr_graph.import_graph(graph);
        //vect_csr_graph.print();

        EdgesArrayNec<int, float, float> weights(vect_csr_graph);
        weights.set_all_random(10);

        ShortestPaths<int, float> sssp_operation(graph);
        float *seq_distances, *vect_csr_distances;
        sssp_operation.allocate_result_memory(graph.get_vertices_count(), &seq_distances);
        sssp_operation.allocate_result_memory(graph.get_vertices_count(), &vect_csr_distances);

        sssp_operation.seq_dijkstra(vect_csr_graph, weights, seq_distances, 0);

        cout << "starting final check!!!! ----------------- " << endl;

        sssp_operation.nec_dijkstra(vect_csr_graph, weights, vect_csr_distances, 0);

        verify_results(seq_distances, vect_csr_distances, vect_csr_graph.get_vertices_count());

        sssp_operation.free_result_memory(seq_distances);
        sssp_operation.free_result_memory(vect_csr_distances);
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
