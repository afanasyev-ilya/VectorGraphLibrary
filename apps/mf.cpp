/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 4.0
#define NEC_VECTOR_CORE_THRESHOLD_VALUE VECTOR_LENGTH
#define COLLECTIVE_FRONTIER_TYPE_CHANGE_THRESHOLD 0.15

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../graph_library.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        // parse args
        AlgorithmCommandOptionsParser parser;
        parser.parse_args(argc, argv);

        // load graph
        ExtendedCSRGraph<int, int> graph;
        EdgesListGraph<int, int> rand_graph;
        if(parser.get_compute_mode() == GENERATE_NEW_GRAPH)
        {
            int vertices_count = pow(2.0, parser.get_scale());
            long long edges_count = vertices_count * parser.get_avg_degree();
            GraphGenerationAPI<int, int>::R_MAT(rand_graph, vertices_count, edges_count, 57, 19, 19, 5, UNDIRECTED_GRAPH);
            graph.import_graph(rand_graph, VERTICES_SORTED, EDGES_SORTED, VECTOR_LENGTH, PULL_TRAVERSAL, MULTIPLE_ARCS_REMOVED);
        }
        else if(parser.get_compute_mode() == LOAD_GRAPH_FROM_FILE)
        {
            double t1 = omp_get_wtime();
            if(!graph.load_from_binary_file(parser.get_graph_file_name()))
                throw "ERROR: graph file not found";
            double t2 = omp_get_wtime();
            cout << "file " << parser.get_graph_file_name() << " loaded in " << t2 - t1 << " sec" << endl;
        }

        int source = 2;
        int sink = 1002;

        MaxFlow<int, int> mf_operation;

        int flow_val = mf_operation.nec_ford_fulkerson(graph, source, sink);
        cout << "flow val: " << flow_val << endl;

        if(parser.get_check_flag())
        {
            int check_flow_val = mf_operation.seq_ford_fulkerson(graph, source, sink);
            cout << "check flow val: " << check_flow_val << endl;
            if(flow_val == check_flow_val)
            {
                cout << "correct!" << endl;
            }
            else
            {
                cout << "error!" << endl;
            }
        }

        //pr_operation.nec_page_rank(graph, page_ranks, 1.0e-4, iterations_count)uil
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
