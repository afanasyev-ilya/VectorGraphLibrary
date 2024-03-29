/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char ** argv)
{
    try
    {
        Parser parser;
        parser.parse_args(argc, argv);

        EdgesContainer edges_container;
        if(parser.get_convert()) // TODO
        {
            cout << get_separators_upper_string("converting " + parser.get_graph_file_name()) << endl;
            GraphGenerationAPI::init_from_txt_file(edges_container, parser.get_convert_name(), DIRECTED_GRAPH);
        }
        else
        {
            cout << get_separators_upper_string("generating " + parser.get_graph_file_name()) << endl;
            Timer tm;
            tm.start();

            int v = pow(2.0, parser.get_scale());
            if(parser.get_synthetic_graph_type() == RMAT)
                GraphGenerationAPI::R_MAT(edges_container, v, v * parser.get_avg_degree(), 57, 19, 19, 5, DIRECTED_GRAPH);
            else if(parser.get_synthetic_graph_type() == RANDOM_UNIFORM)
                GraphGenerationAPI::random_uniform(edges_container, v, v * parser.get_avg_degree(), DIRECTED_GRAPH);

            // if required
            edges_container.random_shuffle_edges();

            tm.end();
            tm.print_time_stats("Generate");
        }

        if(parser.get_graph_storage_format() == EDGES_CONTAINER)
        {
            Timer tm;
            tm.start();
            string full_name = add_extension(parser.get_graph_file_name(), VGL_RUNTIME::select_graph_format(parser));
            edges_container.save_to_binary_file(full_name);
            tm.end();
            tm.print_time_stats("Saving edges container");
        }
        else
        {
            // import to required format
            Timer tm;
            tm.start();
            VGL_Graph out_graph(VGL_RUNTIME::select_graph_format(parser));
            out_graph.import(edges_container);
            tm.end();
            tm.print_time_stats("Import");

            // save graph
            tm.start();
            string full_name = add_extension(parser.get_graph_file_name(), VGL_RUNTIME::select_graph_format(parser));
            out_graph.save_to_binary_file(full_name);
            tm.end();
            tm.print_time_stats("Saving graph file");
        }

        cout << get_separators_bottom_string() << endl;
    }
    catch (const char * error)
    {
        cout << error << endl;
    }
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
