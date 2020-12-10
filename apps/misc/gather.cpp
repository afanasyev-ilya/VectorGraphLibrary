/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define __USE_MULTICORE__

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
        EdgesArray_EL<int> el_weights(el_graph);
        el_weights.set_all_random(10.0);

        VectCSRGraph graph(USE_BOTH);
        graph.import(el_graph);

        GraphAbstractionsMulticore graph_API(graph);
        FrontierMulticore frontier(graph);

        VerticesArray<int> distances(graph);

        graph_API.change_traversal_direction(SCATTER, distances, frontier);

        int inf_val = std::numeric_limits<int>::max() - MAX_WEIGHT;
        auto init_distances = [distances, source_vertex, inf_val] (int src_id, int connections_count, int vector_index)
        {
            if(src_id == source_vertex)
                distances[src_id] = 0;
            else
                distances[src_id] = inf_val;
        };
        frontier.set_all_active();
        graph_API.compute(graph, frontier, init_distances);

        int changes = 0, iterations_count = 0;
        do
        {
            changes = 0;
            iterations_count++;

            auto edge_op_push = [distances, &changes](int src_id, int dst_id, int local_edge_pos,
                                                      long long int global_edge_pos, int vector_index){
                int weight = 1;
                int src_weight = distances[src_id];

                if(distances[dst_id] > src_weight + weight)
                {
                    distances[dst_id] = src_weight + weight;
                    changes = 1;
                }
            };

            graph_API.scatter(graph, frontier, edge_op_push);
        }
        while(changes);
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
