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
        graph.import(el_graph);

        // print size of VectCSR graph
        graph.print_size();

        // create graph weights and set them random
        EdgesArrayNec<int> weights(graph);
        weights.set_all_random(MAX_WEIGHT);

        //graph.print();
        //graph.print_with_weights(weights);

        // run different SSSP algorithms
        VerticesArrayNec<int> push_distances(graph, SCATTER);
        ShortestPaths::nec_dijkstra(graph, weights, push_distances, 0, ALL_ACTIVE, PUSH_TRAVERSAL);

        VerticesArrayNec<int> pull_distances(graph, GATHER);
        ShortestPaths::nec_dijkstra(graph, weights, pull_distances, 0, ALL_ACTIVE, PULL_TRAVERSAL);

        VerticesArrayNec<int> partial_active_distances(graph, SCATTER);
        ShortestPaths::nec_dijkstra(graph, weights, partial_active_distances, 0, PARTIAL_ACTIVE, PUSH_TRAVERSAL);

        // compute reference result
        VerticesArrayNec<int> seq_distances(graph, SCATTER);
        ShortestPaths::seq_dijkstra(graph, weights, seq_distances, 0);

        cout << "push check" << endl;
        verify_results(graph, push_distances, seq_distances);

        cout << "pull check" << endl;
        verify_results(graph, pull_distances, seq_distances);

        cout << "partial check" << endl;
        verify_results(graph, partial_active_distances, seq_distances);

        cout << " ----------------------------- " << endl;
        ShardedGraph sharded_graph;
        sharded_graph.import(el_graph);

        cout << "import done" << endl;

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
