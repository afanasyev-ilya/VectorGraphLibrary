#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class VGL_COMMON_API
{
public:
    static void init_library(int argc, char **argv)
    {

    }

    static void info_message(string _algo_name)
    {
        cout << " ------------ VGL " << _algo_name << " algorithm ----------- " << endl;
    }

    static void prepare_graph(VGL_Graph &_graph, Parser &_parser)
    {
        if(_parser.get_compute_mode() == GENERATE_NEW_GRAPH)
        {
            EdgesContainer edges_container;
            int v = pow(2.0, _parser.get_scale());
            if(_parser.get_graph_type() == RMAT)
                GraphGenerationAPI::R_MAT(edges_container, v, v * _parser.get_avg_degree(), 57, 19, 19, 5, DIRECTED_GRAPH);
            else if(_parser.get_graph_type() == RANDOM_UNIFORM)
                GraphGenerationAPI::random_uniform(edges_container, v, v * _parser.get_avg_degree(), DIRECTED_GRAPH);
            _graph.import(edges_container);
        }
        else if(_parser.get_compute_mode() == LOAD_GRAPH_FROM_FILE)
        {
            /*Timer tm;
            tm.start();
            if(!_graph.load_from_binary_file(_parser.get_graph_file_name()))
                throw "Error: graph file not found";
            tm.end();
            tm.print_time_stats("Graph load");*/
        }
    }

    static void finalize_library()
    {

    }

    static void start_measuring_stats()
    {
        performance_stats.reset_timers();
    }

    static void stop_measuring_stats(long long _edges_count)
    {
        performance_stats.update_timer_stats();
        performance_stats.print_timers_stats();
        performance_stats.print_perf(_edges_count);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
