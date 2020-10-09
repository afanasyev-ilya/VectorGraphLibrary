/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0
#define VECTOR_ENGINE_THRESHOLD_VALUE VECTOR_LENGTH * MAX_SX_AURORA_THREADS * 4096
#define VECTOR_CORE_THRESHOLD_VALUE 5*VECTOR_LENGTH
#define __PRINT_API_PERFORMANCE_STATS__
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
        if(parser.get_graph_type() == RMAT)
            GraphGenerationAPI::R_MAT(el_graph, v, v * parser.get_avg_degree(), 57, 19, 19, 5, DIRECTED_GRAPH);
        else if(parser.get_graph_type() == RANDOM_UNIFORM)
            GraphGenerationAPI::random_uniform(el_graph, v, v * parser.get_avg_degree(), DIRECTED_GRAPH);

        // create vect CSR graph
        VectCSRGraph graph;
        graph.import_graph(el_graph);

        // create graph weights and set them random
        EdgesArrayNec<float> weights(graph);
        weights.set_all_random(MAX_WEIGHT);
        //weights.set_all_constant(1.0);

        //graph.print();
        //graph.print_with_weights(weights);

        // allocate vertices array

        // run different SSSP algorithms
        VerticesArrayNec<float> push_distances(graph, SCATTER);
        ShortestPaths::nec_dijkstra(graph, weights, push_distances, 0, ALL_ACTIVE, PUSH_TRAVERSAL);

        VerticesArrayNec<float> pull_distances(graph, GATHER);
        ShortestPaths::nec_dijkstra(graph, weights, pull_distances, 0, ALL_ACTIVE, PULL_TRAVERSAL);

        VerticesArrayNec<float> partial_active_distances(graph, GATHER);
        ShortestPaths::nec_dijkstra(graph, weights, partial_active_distances, 0, PARTIAL_ACTIVE, PUSH_TRAVERSAL);

        // compute reference result
        VerticesArrayNec<float> seq_distances(graph, SCATTER);
        ShortestPaths::seq_dijkstra(graph, weights, seq_distances, 0);

        // reorder obtained arrays
        graph.reorder_to_original(push_distances);
        graph.reorder_to_original(pull_distances);
        graph.reorder_to_original(seq_distances);
        graph.reorder_to_original(partial_active_distances);

        cout << "push check" << endl;
        verify_results(push_distances.get_ptr(), seq_distances.get_ptr(), graph.get_vertices_count());

        cout << "pull check" << endl;
        verify_results(pull_distances.get_ptr(), seq_distances.get_ptr(), graph.get_vertices_count());

        cout << "partial check" << endl;
        verify_results(partial_active_distances.get_ptr(), seq_distances.get_ptr(), graph.get_vertices_count());
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
