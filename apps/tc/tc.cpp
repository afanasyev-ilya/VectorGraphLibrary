/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 5.0
#define VECTOR_ENGINE_THRESHOLD_VALUE VECTOR_LENGTH*MAX_SX_AURORA_THREADS*128
#define VECTOR_CORE_THRESHOLD_VALUE 3*VECTOR_LENGTH

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void generate_vertex_pairs(VGL_Graph &_graph, vector<pair<int,int>> &_vertex_pairs, int _desired_num_pairs)
{
    int vertices_count = _graph.get_vertices_count();
    for(int i = 0; i < _desired_num_pairs; i++)
    {
        _vertex_pairs.push_back(std::make_pair(rand() % vertices_count, rand() % vertices_count));
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void print_vertex_pairs(vector<pair<int,int>> &_vertex_pairs)
{
    for(auto pair : _vertex_pairs)
    {
        cout << "(" << pair.first << ", " << pair.second << ")" << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        VGL_RUNTIME::init_library(argc, argv);
        VGL_RUNTIME::info_message("Transitive closure (TC)");

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        // prepare graph
        VGL_Graph graph(VGL_RUNTIME::select_graph_format(parser), VGL_RUNTIME::select_graph_optimizations(parser));
        VGL_RUNTIME::prepare_graph(graph, parser);

        // prepare vertex pairs
        int desired_num_pairs = parser.get_number_of_rounds();
        vector<pair<int,int>> vertex_pairs;
        generate_vertex_pairs(graph, vertex_pairs, desired_num_pairs);
        vector<int> answer(vertex_pairs.size());

        // start algorithm
        VGL_RUNTIME::start_measuring_stats();
        VGL_RUNTIME::report_performance(TC::vgl_purdom(graph, vertex_pairs, answer));
        VGL_RUNTIME::stop_measuring_stats(graph.get_edges_count(), parser);

        if(parser.get_check_flag())
        {
            vector<int> check_answer(vertex_pairs.size());
            VGL_RUNTIME::report_performance(TC::vgl_bfs_based(graph, vertex_pairs, check_answer));
            verify_results(&answer[0], &check_answer[0], vertex_pairs.size());
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
