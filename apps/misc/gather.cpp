/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define __USE_NEC_SX_AURORA__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0
#define VECTOR_ENGINE_THRESHOLD_VALUE VECTOR_LENGTH * MAX_SX_AURORA_THREADS * 4096
#define VECTOR_CORE_THRESHOLD_VALUE 5*VECTOR_LENGTH

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        cout << "Gather test..." << endl;

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        // generate random graph
        EdgesListGraph el_graph;
        int v = pow(2.0, parser.get_scale());
        if(parser.get_graph_type() == RMAT)
            GraphGenerationAPI::R_MAT(el_graph, v, v * parser.get_avg_degree(), 57, 19, 19, 5, DIRECTED_GRAPH);
        else if(parser.get_graph_type() == RANDOM_UNIFORM)
            GraphGenerationAPI::random_uniform(el_graph, v, v * parser.get_avg_degree(), DIRECTED_GRAPH);
        el_graph.preprocess_into_csr_based();

        // save graph order
        EdgesListGraph original_graph = el_graph;

        // create EdgesList weights (used in all other tests)
        EdgesArray_EL<int> el_weights(el_graph);
        el_weights.set_all_random(MAX_WEIGHT);

        // EdgesList SSSP
        VerticesArray<int> el_distances(el_graph, ORIGINAL);
        ShortestPaths::nec_dijkstra(el_graph, el_weights, el_distances, 0);

        // create vect CSR graph
        VectCSRGraph graph;
        graph.import(el_graph);

        // print size of VectCSR graph
        graph.print_size();

        // create graph weights and set them random
        EdgesArray_Vect<int> vect_weights(graph);
        vect_weights = el_weights;

        // run different SSSP algorithms
        VerticesArray<int> push_distances(graph, SCATTER);
        ShortestPaths::nec_dijkstra(graph, vect_weights, push_distances, 0, ALL_ACTIVE, PUSH_TRAVERSAL);

        VerticesArray<int> pull_distances(graph, GATHER);
        ShortestPaths::nec_dijkstra(graph, vect_weights, pull_distances, 0, ALL_ACTIVE, PULL_TRAVERSAL);

        VerticesArray<int> partial_active_distances(graph, SCATTER);
        ShortestPaths::nec_dijkstra(graph, vect_weights, partial_active_distances, 0, PARTIAL_ACTIVE, PUSH_TRAVERSAL);

        VerticesArray<int> seq_distances(graph, SCATTER);
        ShortestPaths::seq_dijkstra(graph, vect_weights, seq_distances, 0);

        // obtain original EdgesList graph (since it could be changed during vectCSR generation)
        el_graph = original_graph;

        // sharded test
        ShardedCSRGraph sharded_graph;
        sharded_graph.import(el_graph);

        EdgesArray_Sharded<int> sharded_weights(sharded_graph);
        sharded_weights = el_weights;

        // ShardedGraph SSSP
        VerticesArray<int> sharded_distances(sharded_graph, ORIGINAL);
        performance_stats.reset_timers();
        ShortestPaths::nec_dijkstra(sharded_graph, sharded_weights, sharded_distances, 0);
        performance_stats.print_timers_stats();

        // checks
        cout << "push check" << endl;
        verify_results(graph, push_distances, seq_distances);

        cout << "pull check" << endl;
        verify_results(graph, pull_distances, seq_distances);

        cout << "partial check" << endl;
        verify_results(graph, partial_active_distances, seq_distances);

        cout << "edges list check" << endl;
        verify_results(graph, el_distances, seq_distances);

        cout << "sharded check" << endl;
        verify_results(graph, sharded_distances, push_distances);
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
