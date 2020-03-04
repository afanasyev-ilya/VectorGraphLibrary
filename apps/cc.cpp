//
//  cc.cpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 07/09/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#include "../graph_library.h"
#include <iostream>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        cout << "CC (connected components) test..." << endl;

        // parse args
        AlgorithmCommandOptionsParser parser;
        parser.parse_args(argc, argv);

        // load graph
        ExtendedCSRGraph<int, float> graph;
        EdgesListGraph<int, float> rand_graph;
        if(parser.get_compute_mode() == GENERATE_NEW_GRAPH)
        {
            int vertices_count = pow(2.0, parser.get_scale());
            long long edges_count = vertices_count * parser.get_avg_degree();
            GraphGenerationAPI<int, float>::R_MAT(rand_graph, vertices_count, edges_count, 57, 19, 19, 5, UNDIRECTED_GRAPH);
            graph.import_graph(rand_graph);
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

        ConnectedComponents<int, float> cc_operation(graph);

        #if defined(__USE_NEC_SX_AURORA__) || defined( __USE_INTEL__)
        int *components;
        cc_operation.allocate_result_memory(graph.get_vertices_count(), &components);
        #endif

        #ifdef __USE_NEC_SX_AURORA__
        cc_operation.nec_shiloach_vishkin(graph, components);
        #endif

        #ifdef __USE_NEC_SX_AURORA__
        cc_operation.nec_bfs_based(graph, components);
        #endif

        #ifdef __USE_NEC_SX_AURORA__
        cc_operation.nec_random_mate(graph, components);
        #endif

        if(parser.get_check_flag())
        {
            int *check_components;
            cc_operation.allocate_result_memory(graph.get_vertices_count(), &check_components);
            cc_operation.seq_bfs_based(graph, check_components);

            cc_operation.free_result_memory(check_components);
        }

        #ifdef __USE_NEC_SX_AURORA_TSUBASA__
        cc_operation.free_result_memory(bfs_result);
        #endif
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

