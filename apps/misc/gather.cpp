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
        int source_vertex = rand()%10;

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

        VectCSRGraph graph(USE_BOTH);
        //graph.load_from_binary_file(parser.get_graph_file_name());
        graph.import(el_graph);
        graph.print_size();
        EdgesArray_Vect<int> weights(graph);
        weights.set_all_constant(1.0);

        cout << "both done" << endl;

        VectCSRGraph scatter_graph(USE_SCATTER_ONLY); // TODO USE_VE_ONLY
        //scatter_graph.load_from_binary_file(parser.get_graph_file_name());
        scatter_graph.import(el_graph);
        scatter_graph.print_size();

        cout << "scatter done" << endl;

        VectCSRGraph gather_graph(USE_GATHER_ONLY); // TODO USE_VE_ONLY
        //gather_graph.load_from_binary_file(parser.get_graph_file_name());
        gather_graph.import(el_graph);
        gather_graph.print_size();

        cout << "gather done" << endl;

        // push
        EdgesArray_Vect<int> scatter_weights(scatter_graph);
        scatter_weights.set_all_constant(1.0);
        VerticesArray<int> push_distances(scatter_graph, SCATTER);
        ShortestPaths::nec_dijkstra(scatter_graph, scatter_weights, push_distances, source_vertex, ALL_ACTIVE, PUSH_TRAVERSAL);
        cout << "push done" << endl;

        // pull
        EdgesArray_Vect<int> gather_weights(gather_graph);
        gather_weights.set_all_constant(1.0);
        VerticesArray<int> pull_distances(gather_graph, GATHER);
        ShortestPaths::nec_dijkstra(gather_graph, gather_weights, pull_distances, source_vertex, ALL_ACTIVE, PULL_TRAVERSAL);
        cout << "pull done" << endl;

        if(parser.get_check_flag())
        {
            VerticesArray<int> seq_distances(graph, SCATTER);
            ShortestPaths::seq_dijkstra(graph, weights, seq_distances, source_vertex);

            // checks
            cout << "push check" << endl;
            verify_results(push_distances, seq_distances, 10);

            cout << "pull check" << endl;
            verify_results(pull_distances, seq_distances, 10);
        }

        // save graph order
        /*EdgesListGraph original_graph = el_graph;

        // create EdgesList weights (used in all other tests)
        EdgesArray_EL<int> el_weights(el_graph);
        el_weights.set_all_random(MAX_WEIGHT);

        // EdgesList SSSP
        VerticesArray<int> el_distances(el_graph, ORIGINAL);
        ShortestPaths::nec_dijkstra(el_graph, el_weights, el_distances, source_vertex);

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
        ShortestPaths::nec_dijkstra(graph, vect_weights, push_distances, source_vertex, ALL_ACTIVE, PUSH_TRAVERSAL);

        VerticesArray<int> pull_distances(graph, GATHER);
        ShortestPaths::nec_dijkstra(graph, vect_weights, pull_distances, source_vertex, ALL_ACTIVE, PULL_TRAVERSAL);

        VerticesArray<int> partial_active_distances(graph, SCATTER);
        ShortestPaths::nec_dijkstra(graph, vect_weights, partial_active_distances, source_vertex, PARTIAL_ACTIVE, PUSH_TRAVERSAL);

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
        ShortestPaths::nec_dijkstra(sharded_graph, sharded_weights, sharded_distances, source_vertex);
        performance_stats.print_timers_stats();

        if(parser.get_check_flag())
        {
            VerticesArray<int> seq_distances(graph, SCATTER);
            ShortestPaths::seq_dijkstra(graph, vect_weights, seq_distances, source_vertex);

            // checks
            cout << "push check" << endl;
            verify_results(push_distances, seq_distances);

            cout << "pull check" << endl;
            verify_results(pull_distances, seq_distances);

            cout << "partial check" << endl;
            verify_results(partial_active_distances, seq_distances);

            cout << "edges list check" << endl;
            verify_results(el_distances, seq_distances);

            cout << "sharded check" << endl;
            verify_results(sharded_distances, push_distances);
        }*/
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
