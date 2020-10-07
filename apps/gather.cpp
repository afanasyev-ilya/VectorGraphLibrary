/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0
#define VECTOR_ENGINE_THRESHOLD_VALUE VECTOR_LENGTH * MAX_SX_AURORA_THREADS * 4096
#define VECTOR_CORE_THRESHOLD_VALUE 5*VECTOR_LENGTH
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
        //GraphGenerationAPI::random_uniform(el_graph, v, v * parser.get_avg_degree(), DIRECTED_GRAPH);
        GraphGenerationAPI::R_MAT(el_graph, v, v * parser.get_avg_degree(), 57, 19, 19, 5, DIRECTED_GRAPH);
        cout << "graph generated!" << endl;

        // create vect CSR graph
        VectCSRGraph graph;
        graph.import_graph(el_graph);

        // create graph weights and set them random
        EdgesArrayNec<float> weights(graph);
        weights.set_all_random(MAX_WEIGHT);

        // allocate vertices array
        VerticesArrayNec<float> seq_distances(graph);
        VerticesArrayNec<float> vect_csr_distances(graph);

        // run SSSP algorithms
        ShortestPaths::seq_dijkstra(graph, weights, seq_distances, 1);
        ShortestPaths::nec_dijkstra(graph, weights, vect_csr_distances, 1);

        // check results
        verify_results(seq_distances.get_ptr(), vect_csr_distances.get_ptr(), graph.get_vertices_count());
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
