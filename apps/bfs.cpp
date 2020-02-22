//
//  bfs.cpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 12/09/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#include <stdio.h>
#include "../graph_library.h"
#include <iostream>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
int non_zero_vertices_count(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    int result = vertices_count;
    for(int i = 0; i < vertices_count; i++)
    {
        if((outgoing_ptrs[i+1]-outgoing_ptrs[i]) == 0)
        {
            result = i;
            break;
        }
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
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
            if(!graph.load_from_binary_file(parser.get_graph_file_name()))
                throw "ERROR: graph file not found";
        }

        BFS<int, float> bfs_operation(graph);

        #if defined(__USE_NEC_SX_AURORA__) || defined( __USE_INTEL__)
        int *bfs_levels;
        bfs_operation.allocate_result_memory(graph.get_vertices_count(), &bfs_levels);
        #endif

        #ifdef __USE_GPU__
        int *device_bfs_levels;
        bfs_operation.allocate_device_result_memory(graph.get_vertices_count(), &device_bfs_levels);
        #endif

        const int source_vertex_num = parser.get_steps_count();
        int non_zero_vertices = non_zero_vertices_count(graph);
        cout << "non-zero count: " << (double)non_zero_vertices/graph.get_vertices_count() << endl;

        #ifdef __USE_GPU__
        graph.move_to_device();
        #endif

        double avg_perf = 0;
        double total_time = 0;
        int vertex_to_check = 0;
        for(int i = 0; i < source_vertex_num; i++)
        {
            vertex_to_check = i;//rand()%(non_zero_vertices - 1);
            cout << "starting from vertex: " << vertex_to_check << endl;

            double t1 = omp_get_wtime();
            #ifdef __USE_NEC_SX_AURORA__
            bfs_operation.nec_top_down(graph, bfs_levels, vertex_to_check);
            #endif

            #ifdef __USE_NEC_SX_AURORA__
            bfs_operation.nec_bottom_up(graph, bfs_levels, vertex_to_check);
            #endif

            #ifdef __USE_GPU__
            bfs_operation.gpu_direction_optimising_BFS(graph, device_bfs_levels, vertex_to_check);
            #endif
            double t2 = omp_get_wtime();
            total_time += t2 - t1;

            avg_perf += graph.get_edges_count() / (source_vertex_num*(t2 - t1)*1e6);
        }
        cout << "total time: " << total_time << " sec" << endl;
        cout << "AVG Performance: " << avg_perf << " MTEPS" << endl << endl;

        #ifdef __USE_GPU__
        graph.move_to_host();
        #endif

        if(parser.get_check_flag())
        {
            #ifdef __USE_GPU__
            int *bfs_result;
            bfs_operation.allocate_result_memory(graph.get_vertices_count(), &bfs_result);
            bfs_operation.copy_result_to_host(bfs_result, device_bfs_result, graph.get_vertices_count());
            #endif

            int *check_levels;
            bfs_operation.allocate_result_memory(graph.get_vertices_count(), &check_levels);
            bfs_operation.seq_top_down(graph, check_levels, vertex_to_check);

            verify_results(bfs_levels, check_levels, graph.get_vertices_count());

            bfs_operation.free_result_memory(check_levels);

            #ifdef __USE_GPU__
            bfs_operation.free_result_memory(bfs_result);
            #endif
        }

        #ifdef __USE_NEC_SX_AURORA_TSUBASA__
        bfs_operation.free_result_memory(bfs_result);
        #endif

        #ifdef __USE_GPU__
        bfs_operation.free_device_result_memory(device_bfs_result);
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
