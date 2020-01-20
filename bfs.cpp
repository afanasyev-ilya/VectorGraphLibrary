//
//  bfs.cpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 12/09/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#include <stdio.h>
#include "graph_library.h"
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void nec_test_performance()
{
    int range = 1024*1024*2;
    int indexes_size = 1024*1024*32;
    int *indexes = new int[indexes_size];
    int *result = new int[indexes_size];
    int *data = new int[range];

    for(int i = 0; i < indexes_size; i++)
    {
        indexes[i] = rand() % range;
    }

    double t1 = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < indexes_size; i += 256)
    {
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        for(int j = 0; j < 256; j++)
        {
            result[i + j] = data[indexes[i + j]];
        }
    }
    double t2 = omp_get_wtime();
    cout << "static: " << (sizeof(int)*3.0 * indexes_size)/((t2 - t1)*1e9) << " GB/s" << endl;

    t1 = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < indexes_size; i += 256)
    {
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        for(int j = 0; j < 256; j++)
        {
            result[i + j] = data[indexes[i + j]];
        }
    }
    t2 = omp_get_wtime();
    cout << "dynamic: " << (sizeof(int)*3.0 * indexes_size)/((t2 - t1)*1e9) << " GB/s" << endl;

    cout << "work per core: " << indexes_size / 256 << endl;

    for(int k = 1; k < 1024*256; k *= 2)
    {
        t1 = omp_get_wtime();
        #pragma omp parallel for schedule(static, k)
        for (int i = 0; i < indexes_size; i += 256)
        {
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for (int j = 0; j < 256; j++) {
                result[i + j] = data[indexes[i + j]];
            }
        }
        t2 = omp_get_wtime();
        cout << k << " stride: " << (sizeof(int) * 3.0 * indexes_size) / ((t2 - t1) * 1e9) << " GB/s" << endl;
    }
    cout << endl;

    for(int k = 256; k < indexes_size; k *= 2)
    {
        t1 = omp_get_wtime();
        #pragma omp parallel for schedule(static, 4)
        for (int i = 0; i < k; i += 256)
        {
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for (int j = 0; j < 256; j++)
            {
                result[i + j] = data[indexes[i + j]];
            }
        }
        t2 = omp_get_wtime();
        cout << "small work of size " << k << " : " << (sizeof(int) * 3.0 * k) / ((t2 - t1) * 1e9) << " GB/s" << endl;
    }

    int works_count = 64*1024;
    int *works = new int[works_count];
    for(int w = 0; w < works_count; w++)
        works[w] = (rand() % (indexes_size - 1024));

    cout << "first works: " << endl;
    for(int w = 0; w < 10; w++)
    {
        cout << works[w] << endl;
    }
    cout << endl;

    for(int k = 1; k <= works_count; k *= 2)
    {
        int work_size = 256;
        t1 = omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (int w = 0; w < k; w++) {
            int start = works[w];
            int end = works[w] + work_size;

            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for (int i = start; i < end; i++) {
                result[i] = data[indexes[i]];
            }
        }
        t2 = omp_get_wtime();
        cout << "work test of work count " << k << " : " << (sizeof(int) * 3.0 * work_size * k) / ((t2 - t1) * 1e9) << " GB/s"
             << endl;
    }

    delete[]works;
    delete[]indexes;
    delete[]data;
    delete[]result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        nec_test_performance();
        return;

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
            //GraphGenerationAPI<int, float>::random_uniform(rand_graph, vertices_count, edges_count, UNDIRECTED_GRAPH);
            GraphGenerationAPI<int, float>::R_MAT(rand_graph, vertices_count, edges_count, 57, 19, 19, 5, UNDIRECTED_GRAPH);
            graph.import_graph(rand_graph, VERTICES_SORTED, EDGES_SORTED, 1, PUSH_TRAVERSAL);
        }
        else if(parser.get_compute_mode() == LOAD_GRAPH_FROM_FILE)
        {
            if(!graph.load_from_binary_file(parser.get_graph_file_name()))
                throw "ERROR: graph file not found";
            
            cout << "Loaded graph with name: " << parser.get_graph_file_name() << endl;
        }

        BFS<int, float> bfs_operation(graph);
        
        #if defined(__USE_NEC_SX_AURORA__) || defined( __USE_INTEL__)
        int *bfs_result;
        bfs_operation.allocate_result_memory(graph.get_vertices_count(), &bfs_result);
        #endif
        
        #ifdef __USE_GPU__
        int *device_bfs_result;
        bfs_operation.allocate_device_result_memory(graph.get_vertices_count(), &device_bfs_result);
        #endif
        
        const int source_vertex_num = 5;
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
            vertex_to_check = i+1;//rand()%(non_zero_vertices - 1);
            cout << "starting from vertex: " << vertex_to_check << endl;

            double t1 = omp_get_wtime();
            #ifdef __USE_NEC_SX_AURORA__
            bfs_operation.nec_direction_optimising_BFS(graph, bfs_result, vertex_to_check);
            #endif
            
            #ifdef __USE_GPU__
            bfs_operation.gpu_direction_optimising_BFS(graph, device_bfs_result, vertex_to_check);
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
            
            bfs_operation.verifier(graph, vertex_to_check, bfs_result);
            
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
