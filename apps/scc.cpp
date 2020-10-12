/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 4.0
#define VECTOR_CORE_THRESHOLD_VALUE 4.0*VECTOR_LENGTH
#define COLLECTIVE_FRONTIER_TYPE_CHANGE_THRESHOLD 0.35
#define __PRINT_SAMPLES_PERFORMANCE_STATS__
//#define __PRINT_API_PERFORMANCE_STATS__

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
        VectCSRGraph graph;
        if(parser.get_compute_mode() == GENERATE_NEW_GRAPH)
        {
            EdgesListGraph el_graph;
            int v = pow(2.0, parser.get_scale());
            if(parser.get_graph_type() == RMAT)
                GraphGenerationAPI::R_MAT(el_graph, v, v * parser.get_avg_degree(), 57, 19, 19, 5, DIRECTED_GRAPH);
            else if(parser.get_graph_type() == RANDOM_UNIFORM)
                GraphGenerationAPI::random_uniform(el_graph, v, v * parser.get_avg_degree(), DIRECTED_GRAPH);
            graph.import(el_graph);
        }
        else if(parser.get_compute_mode() == LOAD_GRAPH_FROM_FILE)
        {
            double t1 = omp_get_wtime();
            if(!graph.load_from_binary_file(parser.get_graph_file_name()))
                throw "Error: graph file not found";
            double t2 = omp_get_wtime();
            cout << "file " << parser.get_graph_file_name() << " loaded in " << t2 - t1 << " sec" << endl;
        }

        // print size of VectCSR graph
        graph.print_size();

        // compute SCC
        cout << "SCC computations started..." << endl;

        VerticesArrayNec<int> components(graph, SCATTER); // TODO selection for DO/BU

        #ifdef __PRINT_API_PERFORMANCE_STATS__
        PerformanceStats::reset_API_performance_timers();
        #endif

        #ifdef __USE_NEC_SX_AURORA__
        SCC::nec_forward_backward(graph, components);
        #endif

        #ifdef __PRINT_API_PERFORMANCE_STATS__
        PerformanceStats::print_API_performance_timers(graph.get_edges_count());
        #endif

        // check if required
        if(parser.get_check_flag())
        {
            VerticesArrayNec<int> check_components(graph, SCATTER);
            SCC::seq_tarjan(graph, check_components);
            equal_components(graph, components, check_components);
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
