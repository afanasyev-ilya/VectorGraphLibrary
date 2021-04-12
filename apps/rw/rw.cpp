/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define COMPUTE_INT_ELEMENTS 10.0 // or 8?
#define FRONTIER_TYPE_CHANGE_THRESHOLD 1.0

#define __PRINT_API_PERFORMANCE_STATS__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        cout << "RW (Random Walks) test..." << endl;

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        VectCSRGraph graph;
        if(parser.get_compute_mode() == GENERATE_NEW_GRAPH)
        {
            EdgesListGraph el_graph;
            int v = pow(2.0, parser.get_scale());
            if(parser.get_graph_type() == RMAT)
                GraphGenerationAPI::R_MAT(el_graph, v, v * parser.get_avg_degree(), 57, 19, 19, 5, UNDIRECTED_GRAPH);
            else if(parser.get_graph_type() == RANDOM_UNIFORM)
                GraphGenerationAPI::random_uniform(el_graph, v, v * parser.get_avg_degree(), UNDIRECTED_GRAPH);
            graph.import(el_graph);
        }
        else if(parser.get_compute_mode() == LOAD_GRAPH_FROM_FILE)
        {
            Timer tm;
            tm.start();
            if(!graph.load_from_binary_file(parser.get_graph_file_name()))
                throw "Error: graph file not found";
            tm.end();
            tm.print_time_stats("Graph load");
        }

        // do calculations
        cout << "Computations started..." << endl;
        cout << "Running RW algorithm " << endl;
        int walk_vertices_num = parser.get_walk_vertices_percent() * (graph.get_vertices_count()/100.0);
        int walk_length = parser.get_number_of_rounds();
        cout << "walk vertices num: " << walk_vertices_num << endl;
        cout << "walk length: " << walk_length << endl;

        // generate list of walk vertices
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

        // do random walks
        VerticesArray<int> walk_results(graph);
        performance_stats.reset_timers();
        RW::vgl_random_walk(graph, walk_vertices, walk_vertices_num, walk_length, walk_results);
        performance_stats.update_timer_stats();
        performance_stats.print_timers_stats();
        performance_stats.print_perf(graph.get_edges_count());

        // run sequential algorithm for timing comparison
        if(parser.get_check_flag())
        {
            VerticesArray<int> check_walk_results(graph);
            RW::seq_random_walk(graph, walk_vertices, walk_vertices_num, walk_length, check_walk_results);
            cout << "since walks are random it is not possible to check" << endl;
        }
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
