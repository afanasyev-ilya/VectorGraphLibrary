/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 4.0
#define __PRINT_API_PERFORMANCE_STATS__
#define __PRINT_SAMPLES_PERFORMANCE_STATS__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../graph_library.h"
#include <iostream>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        cout << "Label Propagation test..." << endl;

        // parse args
        AlgorithmCommandOptionsParser parser;
        parser.parse_args(argc, argv);

        //VectorisedCSRGraph<int, float> graph;
        ExtendedCSRGraph<int, float> graph;
        EdgesListGraph<int, float> rand_graph;
        if(parser.get_compute_mode() == GENERATE_NEW_GRAPH)
        {
            int vertices_count = pow(2.0, parser.get_scale());
            long long edges_count = vertices_count * parser.get_avg_degree();
            //GraphGenerationAPI<int, float>::random_uniform(rand_graph, vertices_count, edges_count, UNDIRECTED_GRAPH);
            GraphGenerationAPI<int, float>::R_MAT(rand_graph, vertices_count, edges_count, 57, 19, 19, 5, UNDIRECTED_GRAPH);
            graph.import_graph(rand_graph, VERTICES_SORTED, EDGES_SORTED, VECTOR_LENGTH, PULL_TRAVERSAL);
        }
        else if(parser.get_compute_mode() == LOAD_GRAPH_FROM_FILE)
        {
            double t1 = omp_get_wtime();
            if(!graph.load_from_binary_file(parser.get_graph_file_name()))
                throw "ERROR: graph file not found";
            double t2 = omp_get_wtime();
            cout << "file " << parser.get_graph_file_name() << " loaded in " << t2 - t1 << " sec" << endl;
        }

        GraphAnalytics graph_analytics;
        graph_analytics.analyse_graph_stats(graph, parser.get_graph_file_name());

        // compute SSSP
        int last_src_vertex = 0;
        cout << "Computations started..." << endl;
        LabelPropagation<int, float> lp_operation;

        int *labels;
        lp_operation.allocate_result_memory(graph.get_vertices_count(), &labels);

        #ifdef __USE_GPU__
        graph.move_to_device();
        #endif

        #ifdef __USE_GPU__
        lp_operation.gpu_lp(graph, labels);
        #endif

        #ifdef __USE_GPU__
        graph.move_to_host();
        #endif

        lp_operation.free_result_memory(labels);
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
