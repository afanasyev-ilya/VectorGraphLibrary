/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define COMPUTE_INT_ELEMENTS 10.0 // or 8?
#define FRONTIER_TYPE_CHANGE_THRESHOLD 1.0

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        VGL_RUNTIME::init_library(argc, argv);
        VGL_RUNTIME::info_message("RW");

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        // prepare graph
        VGL_Graph graph(VGL_RUNTIME::select_graph_format(parser), VGL_RUNTIME::select_graph_optimizations(parser));
        VGL_RUNTIME::prepare_graph(graph, parser, UNDIRECTED_GRAPH);

        // generate list of walk vertices
        int walk_vertices_num = parser.get_walk_vertices_percent() * (graph.get_vertices_count()/100.0);
        int walk_length = parser.get_number_of_rounds();
        cout << "walk vertices num: " << walk_vertices_num << endl;
        cout << "walk length: " << walk_length << endl;
        Timer tm;
        vector<int> walk_vertices;
        for(int i = 0; i < graph.get_vertices_count(); i++)
        {
            int prob = rand() % 100;
            if(prob < parser.get_walk_vertices_percent())
                walk_vertices.push_back(i);
        }
        tm.end();
        tm.print_time_stats("generate list of walk vertices");

        // run algorithm
        VerticesArray<int> walk_results(graph);
        VGL_RUNTIME::start_measuring_stats();
        RW::vgl_random_walk(graph, walk_vertices, walk_vertices_num, walk_length, walk_results);
        VGL_RUNTIME::stop_measuring_stats(graph.get_edges_count(), parser);

        if(parser.get_check_flag())
        {
            //VerticesArray<int> check_walk_results(graph);
            //RW::seq_random_walk(graph, walk_vertices, walk_vertices_num, walk_length, check_walk_results);
            cout << "since walks are random it is not possible to check" << endl;
        }

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
