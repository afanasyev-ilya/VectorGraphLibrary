/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0
#define VECTOR_ENGINE_THRESHOLD_VALUE VECTOR_LENGTH*MAX_SX_AURORA_THREADS*128
#define VECTOR_CORE_THRESHOLD_VALUE 3*VECTOR_LENGTH

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        VGL_RUNTIME::init_library(argc, argv);
        VGL_RUNTIME::info_message("Maximum Flow (MF)");

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        // prepare graph
        VGL_Graph graph(VGL_RUNTIME::select_graph_format(parser), VGL_RUNTIME::select_graph_optimizations(parser));
        VGL_RUNTIME::prepare_graph(graph, parser);

        // init flows
        EdgesArray<int> flows(graph);

        // start algorithm
        VGL_RUNTIME::start_measuring_stats();
        double avg_perf = 0;
        for(int i = 0; i < parser.get_number_of_rounds(); i++)
        {
            // clear flows before each iteration
            flows.set_all_constant(MAX_WEIGHT);

            int max_flow_val = 0;
            int source = graph.select_random_nz_vertex(Parser::convert_traversal_type(parser.get_traversal_direction()));
            int sink = graph.select_random_nz_vertex(Parser::convert_traversal_type(parser.get_traversal_direction()));

            avg_perf += MF::vgl_ford_fulkerson(graph, flows, source, sink, max_flow_val) / parser.get_number_of_rounds();
            cout << "Result: " << max_flow_val << endl;

            if(parser.get_check_flag())
            {
                flows.set_all_constant(MAX_WEIGHT);

                int check_flow = 0;
                MF::seq_ford_fulkerson(graph, flows, source, sink, check_flow);
                cout << max_flow_val << " vs " << check_flow << endl;
                if(max_flow_val == check_flow)
                    cout << "Results are equal" << endl;
                else
                    cout << "Results are NOT equal, error_count = " << graph.get_vertices_count() << endl;

                if(graph.get_vertices_count() < 32)
                {
                    EdgesArray<int> original_weights(graph);
                    original_weights.set_all_constant(MAX_WEIGHT);
                    save_flows_to_graphviz_file(graph, original_weights, flows, source, sink, "mf_vis.txt");
                }
            }
        }
        VGL_RUNTIME::stop_measuring_stats(graph.get_edges_count(), parser);
        VGL_RUNTIME::report_performance(avg_perf);

        VGL_RUNTIME::finalize_library();
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
