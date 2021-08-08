#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VGL_RUNTIME::init_library(int argc, char **argv)
{
    vgl_library_data.init(argc, argv);

    #ifdef __USE_ASL__
    ASL_CALL(asl_library_initialize());
    #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VGL_RUNTIME::info_message(string _algo_name)
{
    cout << " ------------ VGL " << _algo_name << " algorithm ----------- " << endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VGL_RUNTIME::prepare_graph(VGL_Graph &_graph, Parser &_parser, DirectionType _direction)
{
    if(_parser.get_compute_mode() == GENERATE_NEW_GRAPH)
    {
        Timer tm;
        tm.start();
        EdgesContainer edges_container;
        int v = pow(2.0, _parser.get_scale());
        if(_parser.get_graph_type() == RMAT)
            GraphGenerationAPI::R_MAT(edges_container, v, v * _parser.get_avg_degree(), 57, 19, 19, 5, _direction);
        else if(_parser.get_graph_type() == RANDOM_UNIFORM)
            GraphGenerationAPI::random_uniform(edges_container, v, v * _parser.get_avg_degree(), _direction);
        tm.end();
        tm.print_time_stats("graph generation");

        tm.start();
        edges_container.random_shuffle_edges();
        tm.end();
        tm.print_time_stats("random_shuffle");

        tm.start();
        _graph.import(edges_container);
        tm.end();
        tm.print_time_stats("import graph");
    }
    else if(_parser.get_compute_mode() == LOAD_GRAPH_FROM_FILE)
    {
        Timer tm;
        tm.start();
        if(!_graph.load_from_binary_file(_parser.get_graph_file_name()))
            throw "Error: graph file not found";
        tm.end();
        tm.print_time_stats("Graph load");
    }

    #ifdef __USE_MPI__
    vgl_library_data.allocate_exchange_buffers(_graph.get_vertices_count(), sizeof(double));
    #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GraphType VGL_RUNTIME::select_graph_format(Parser &_parser)
{
    return _parser.get_graph_storage_format();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VGL_RUNTIME::finalize_library()
{
    #ifdef __USE_ASL__
    ASL_CALL(asl_library_finalize());
    #endif

    vgl_library_data.finalize();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VGL_RUNTIME::start_measuring_stats()
{
    performance_stats.reset_timers();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VGL_RUNTIME::stop_measuring_stats(long long _edges_count, Parser &_parser)
{
    performance_stats.update_timer_stats();
    performance_stats.print_timers_stats();
    performance_stats.print_perf(_edges_count, _parser.get_number_of_rounds());
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vgl_runtime.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

