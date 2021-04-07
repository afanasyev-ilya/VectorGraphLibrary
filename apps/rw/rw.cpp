/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 1.0

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
        int walk_vertices_num = parser.get_walk_vertices_num();
        int walk_length = parser.get_walk_lengths();

        // generate list of walk vertices
        Timer tm;
        std::set<int> walk_vertices;
        for(int i = 0; i < walk_vertices_num; i++)
        {
            while(true)
            {
                int vertex_id = rand()% (graph.get_vertices_count());
                if (!walk_vertices.count(vertex_id))
                {
                    walk_vertices.insert(vertex_id);
                    break;
                }
            }
        }
        tm.end();
        tm.print_time_stats("generate list of walk vertices");

        // store walk paths if required
        int *walk_paths = NULL;
        if(parser.get_store_walk_paths())
            MemoryAPI::allocate_array(&walk_paths, walk_vertices_num * walk_length);

        // do random walks
        VerticesArray<int> walk_results(graph);
        RW::vgl_random_walk(graph, walk_vertices, walk_vertices_num, walk_length, walk_results, walk_paths);

        if(parser.get_store_walk_paths())
            MemoryAPI::free_array(walk_paths);

        performance_stats.print_perf(graph.get_edges_count());
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
