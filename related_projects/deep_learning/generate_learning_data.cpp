/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"
#include "prepare_nn_input.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double measure_perf(VGL_Graph &_graph, Parser &_parser)
{
    double avg_perf = 0;

    VerticesArray<float> distances(_graph, Parser::convert_traversal_type(_parser.get_traversal_direction()));
    EdgesArray<float> weights(_graph);
    weights.set_all_random(MAX_WEIGHT);

    // start algorithm
    VGL_RUNTIME::start_measuring_stats();
    for(int i = 0; i < 20; i++)
    {
        int source_vertex = _graph.select_random_nz_vertex(Parser::convert_traversal_type(_parser.get_traversal_direction()));
        ShortestPaths::vgl_dijkstra(_graph, weights, distances, source_vertex,
                                    _parser.get_algorithm_frontier_type(),
                                    _parser.get_traversal_direction());
    }
    VGL_RUNTIME::stop_measuring_stats(_graph.get_edges_count(), _parser);

    avg_perf = performance_stats.get_avg_perf(_graph.get_edges_count());
    cout << "avg perf: " << avg_perf << endl;
    return avg_perf;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct ForOpt
{
    GraphStorageFormat format;
    GraphStorageOptimizations optimization;
    ForOpt(GraphStorageFormat _format, GraphStorageOptimizations _optimization)
    {
        format = _format;
        optimization = _optimization;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void make_data_folders(const int _num_formats)
{
    for(int i = 0; i < _num_formats; i++)
        mkdir(std::to_string(i).c_str(), 0777);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int main(int argc, char **argv)
{
    try
    {
        VGL_RUNTIME::init_library(argc, argv);

        // parse args
        Parser parser;
        parser.parse_args(argc, argv);

        const int num_formats = 6;

        ForOpt run_data[num_formats] = {
                ForOpt(VECTOR_CSR_GRAPH, OPT_NONE),
                ForOpt(CSR_GRAPH, OPT_NONE),
                ForOpt(CSR_VG_GRAPH, OPT_NONE),
                ForOpt(EDGES_LIST_GRAPH, OPT_NONE),
                ForOpt(EDGES_LIST_GRAPH, EL_2D_SEGMENTED),
                ForOpt(EDGES_LIST_GRAPH, EL_CSR_BASED)};

        make_data_folders(num_formats);

        EdgesContainer graph_container;

        string out_file_name = "";
        if(parser.get_compute_mode() == GENERATE_NEW_GRAPH)
        {
            string type;
            int v = pow(2.0, parser.get_scale());
            if(parser.get_synthetic_graph_type() == RMAT)
            {
                type = "rmat";
                GraphGenerationAPI::R_MAT(graph_container, v, v * parser.get_avg_degree(), 57, 19, 19, 5, DIRECTED_GRAPH);
            }
            else if (parser.get_synthetic_graph_type() == RANDOM_UNIFORM)
            {
                type = "ru";
                GraphGenerationAPI::random_uniform(graph_container, v, v * parser.get_avg_degree(), DIRECTED_GRAPH);
            }

            graph_container.random_shuffle_edges();
            out_file_name += type + "_syn_" + std::to_string(parser.get_scale()) + "_" + std::to_string(parser.get_avg_degree());
        }

        double best_perf = 0;
        int best_run = 0;
        for(int i = 0; i < num_formats; i++)
        {
            VGL_Graph graph(run_data[i].format, run_data[i].optimization);
            graph.import(graph_container);

            double perf = measure_perf(graph, parser);
            if(perf > best_perf)
            {
                best_perf = perf;
                best_run = i;
            }
        }

        cout << "BEST PERF " << best_perf << " ON: " << run_data[best_run].format << " " <<
            run_data[best_run].optimization << endl;

        double* nn_input = convert_graph_to_nn_input(graph_container);
        save_nn_input_to_file(nn_input, best_run, out_file_name);

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
