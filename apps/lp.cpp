/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 4.0

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
        UndirectedCSRGraph<int, float> graph;
        EdgesListGraph<int, float> rand_graph;
        if(parser.get_compute_mode() == GENERATE_NEW_GRAPH)
        {
            int vertices_count = pow(2.0, parser.get_scale());
            long long edges_count = vertices_count * parser.get_avg_degree();
            //GraphGenerationAPI<int, float>::random_uniform(rand_graph, vertices_count, edges_count, UNDIRECTED_GRAPH);
            GraphGenerationAPI<int, float>::R_MAT(rand_graph, vertices_count, edges_count, 57, 19, 19, 5, DIRECTED_GRAPH);
            graph.import_graph(rand_graph, VERTICES_SORTED, EDGES_RANDOM_SHUFFLED, VECTOR_LENGTH, PULL_TRAVERSAL, MULTIPLE_ARCS_PRESENT);
        }
        else if(parser.get_compute_mode() == LOAD_GRAPH_FROM_FILE)
        {
            double t1 = omp_get_wtime();
            if(!graph.load_from_binary_file(parser.get_graph_file_name()))
                throw "Error: graph file not found";
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
        cout << "Always active test" << endl;
        lp_operation.gpu_lp(graph, labels, AlwaysActive);

        cout << "Active-passive-inner-condition test" << endl;
        lp_operation.gpu_lp(graph, labels, ActivePassiveInner);

        /*cout << "Label Changed on previous iteration test" << endl;
        lp_operation.gpu_lp(graph, labels, LabelChangedOnPreviousIteration);

        cout << "Label Changed Recently test" << endl;
        lp_operation.gpu_lp(graph, labels, LabelChangedRecently);*/
        #endif

        #ifdef __USE_GPU__
        graph.move_to_host();
        #endif

        if(parser.get_check_flag())
        {
            lp_operation.seq_lp(graph, labels);
        }

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
