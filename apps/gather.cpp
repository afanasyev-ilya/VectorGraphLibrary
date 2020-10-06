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

void print_first_edges(EdgesListGraph &_rand_graph)
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

        // generate random graph
        EdgesListGraph el_graph;
        int v = pow(2.0, parser.get_scale());
        GraphGenerationAPI::random_uniform(el_graph, v, v * parser.get_avg_degree(), DIRECTED_GRAPH);
        cout << "graph generated!" << endl;

        // preprocess it
        el_graph.preprocess_into_csr_based();
        el_graph.print_in_csr_format();

        // create vect CSR graph
        VectCSRGraph graph;
        graph.import_graph(el_graph);
        graph.print();

        // create graph weights and set them random
        EdgesArrayNec<float> weights(graph);
        weights.set_all_random(MAX_WEIGHT);

        /*ShortestPaths sssp_operation(graph);
        float *seq_distances, *vect_csr_distances;
        sssp_operation.allocate_result_memory(graph.get_vertices_count(), &seq_distances);
        sssp_operation.allocate_result_memory(graph.get_vertices_count(), &vect_csr_distances);

        sssp_operation.seq_dijkstra(vect_csr_graph, weights, seq_distances, 0);

        cout << "starting final check!!!! ----------------- " << endl;

        sssp_operation.nec_dijkstra(vect_csr_graph, weights, vect_csr_distances, 0);

        verify_results(seq_distances, vect_csr_distances, vect_csr_graph.get_vertices_count());

        sssp_operation.free_result_memory(seq_distances);
        sssp_operation.free_result_memory(vect_csr_distances);*/
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
