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
    int range = 512*1024;
    int indexes_size = 1024*1024*32;
    int *indexes = new int[indexes_size];
    int *result = new int[indexes_size];
    int *data = new int[range];

    for(int i = 0; i < indexes_size; i++)
    {
        indexes[i] = rand() % range;
    }

    #pragma omp parallel
    {};

    int scalar = 20;

    auto sum_op = [scalar](int el1, int el2) {
        return scalar*el1 + el2;
    };

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
            result[i + j] = scalar*indexes[i + j] + indexes[i + j];
        }
    }
    double t2 = omp_get_wtime();
    cout << "NO lambda: " << (sizeof(int)*2.0 * indexes_size)/((t2 - t1)*1e9) << " GB/s" << endl;

    for(int i = 0; i < 10; i++)
        cout << result[i] << endl;

    t1 = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < indexes_size; i += 256)
    {
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        for(int j = 0; j < 256; j++)
        {
            result[i + j] = sum_op(indexes[i + j], indexes[i + j]);
        }
    }
    t2 = omp_get_wtime();
    cout << "with lambda: " << (sizeof(int)*2.0 * indexes_size)/((t2 - t1)*1e9) << " GB/s" << endl;

    for(int i = 0; i < 10; i++)
        cout << result[i] << endl;

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

    int works_count = 32*1024;
    int *works = new int[works_count];
    int *work_sizes = new int[works_count];
    for(int w = 0; w < works_count; w++)
    {
        //int min_size = 1024;
        //int max_size = 2048;
        works[w] = (rand() % (indexes_size - 1024));
        work_sizes[w] = 512 + rand()%(1024-512);
    }

    cout << "first works: " << endl;
    for(int w = 0; w < 10; w++)
    {
        cout << works[w] << endl;
    }
    cout << endl;

    for(int k = 1; k < works_count; k *= 2)
    {
        //int work_size = 512;
        t1 = omp_get_wtime();
        #pragma omp parallel for schedule(static, 2)
        for (int w = 0; w < k; w++)
        {
            int start = works[w];
            int end = works[w] + work_sizes[w];

            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for (int i = start; i < end; i++)
            {
                result[i] = data[indexes[i]];
            }
        }
        t2 = omp_get_wtime();
        double work_len = 0;
        for(int w = 0; w < k; w++)
            work_len += work_sizes[w];
        cout << "work test of work count " << k << " : " << (sizeof(int) * 3.0 * work_len) / ((t2 - t1) * 1e9) << " GB/s"
             << endl;
    }

    delete[]works;
    delete[]work_sizes;
    delete[]indexes;
    delete[]data;
    delete[]result;
}

template <typename _TVertexValue, typename _TEdgeWeight>
void traversal_test(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph)
{
    cout << "graph trav test" << endl;
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    int *data = new int[vertices_count];
    int *result = new int[edges_count];

    int *frontier_ids = new int[vertices_count];
    for(int i = 0; i < vertices_count; i++)
        frontier_ids[i] = i;

    int *cached_data = _graph.template allocate_private_caches<int>(8);

    #pragma omp parallel
    {}

    int large_threshold_size = 256*1024;
    int medium_threshold_size = 256;

    // split graphs into parts
    int large_threshold_vertex = 0;
    int medium_threshold_vertex = 0;
    for(int src_id = 0; src_id < vertices_count - 1; src_id++)
    {
        int cur_size = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
        int next_size = 0;
        if(src_id < (vertices_count - 2))
        {
            next_size = outgoing_ptrs[src_id + 2] - outgoing_ptrs[src_id + 1];
        }
        if((cur_size >= large_threshold_size) && (next_size < large_threshold_size))
        {
            large_threshold_vertex = src_id;
        }

        if((cur_size >= medium_threshold_size) && (next_size < medium_threshold_size))
        {
            medium_threshold_size = src_id;
        }
    }

    double t1 = omp_get_wtime();
    for(int front_pos = 0; front_pos < large_threshold_vertex; front_pos++)
    {
        int src_id = frontier_ids[front_pos];
        long long int start = outgoing_ptrs[src_id];
        long long int end = outgoing_ptrs[src_id + 1];

        #pragma omp parallel
        {
            int *private_data = _graph.template get_private_data_pointer<int>(cached_data);

            #pragma omp for schedule(static)
            for (int i = start; i < end; i += VECTOR_LENGTH)
            {
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                for (int j = 0; j < VECTOR_LENGTH; j++)
                {
                    int dst_id = outgoing_ids[i + j];
                    result[i + j] = _graph.template load_vertex_data_cached<int>(dst_id, data, private_data);
                }
            }
        }
    }
    double t2 = omp_get_wtime();
    double wall_time = t2 - t1;
    double work = outgoing_ptrs[large_threshold_vertex];
    cout << "work: " << (int)work << endl;
    cout << "time: " << (t2 - t1)*1000 << " ms" << endl;
    cout << "part 1(large) BW " << " : " << ((sizeof(int)*3.0) * work) / ((t2 - t1) * 1e9) << " GB/s" << endl << endl;

    t1 = omp_get_wtime();
    #pragma omp parallel
    {
        int *private_data = _graph.template get_private_data_pointer<int>(cached_data);

        #pragma omp for schedule(static, 4)
        for (int front_pos = large_threshold_vertex; front_pos < medium_threshold_size; front_pos++)
        {
            int src_id = front_pos;//frontier_ids[front_pos];
            long long int start = outgoing_ptrs[src_id];
            long long int end = outgoing_ptrs[src_id + 1];

            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for (int i = start; i < end; i++)
            {
                int dst_id = outgoing_ids[i];
                result[i] = _graph.template load_vertex_data_cached<int>(dst_id, data, private_data);
            }
        }
    }
    t2 = omp_get_wtime();
    wall_time += t2 - t1;
    work = outgoing_ptrs[medium_threshold_size] - outgoing_ptrs[large_threshold_vertex];
    cout << "time: " << (t2 - t1)*1000 << " ms" << endl;
    cout << "part 2(medium) BW " << " : " << ((sizeof(int)*3.0) * work) / ((t2 - t1) * 1e9) << " GB/s" << endl << endl;

    t1 = omp_get_wtime();
    #pragma omp parallel
    {
        int *private_data = _graph.template get_private_data_pointer<int>(cached_data);

        #pragma omp for schedule(static, 4)
        for(int src_id = medium_threshold_size; src_id < vertices_count; src_id += VECTOR_LENGTH)
        {
            long long starts[VECTOR_LENGTH];
            long long ends[VECTOR_LENGTH];
            long long int start = outgoing_ptrs[src_id];
            long long int end = outgoing_ptrs[src_id + 1];

            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            int max_size = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                starts[i] = outgoing_ptrs[src_id + i];
                ends[i] = outgoing_ptrs[src_id + i + 1];
            }

            for(int edge_pos = 0; edge_pos < max_size; edge_pos++)
            {
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    if(edge_pos < ends[i])
                    {
                        int dst_id = outgoing_ids[starts[i] + edge_pos];
                        result[starts[i] + edge_pos] = _graph.template load_vertex_data_cached<int>(dst_id, data,
                                                                                                    private_data);
                    }
                }
            }
        }
    }
    t2 = omp_get_wtime();
    double first_part_time = wall_time;
    wall_time += t2 - t1;
    work = outgoing_ptrs[vertices_count] - outgoing_ptrs[medium_threshold_size];
    cout << "time: " << (t2 - t1)*1000 << " ms" << endl;
    cout << "part 3(small) BW " << " : " << ((sizeof(int)*3.0) * work) / ((t2 - t1) * 1e9) << " GB/s" << endl << endl;
    cout << "wall BW: " << ((sizeof(int)*3.0) * edges_count) / (wall_time * 1e9) << " GB/s" << endl;
    cout << "no reminder BW: " << ((sizeof(int)*3.0) * outgoing_ptrs[medium_threshold_size]) / (first_part_time * 1e9) << " GB/s" << endl;
    cout << outgoing_ptrs[large_threshold_size] << " " << outgoing_ptrs[medium_threshold_size] << " " << outgoing_ptrs[vertices_count] << endl;

    cout << "large count: " << large_threshold_vertex << " | " << 100.0*(large_threshold_vertex)/vertices_count << endl;
    cout << "medium_count: " << medium_threshold_size - large_threshold_vertex << " | " << 100.0*(medium_threshold_size - large_threshold_vertex)/vertices_count << endl;
    cout << "small count: " << vertices_count - medium_threshold_size << " | " << 100.0*(vertices_count - medium_threshold_size)/vertices_count << endl;

    delete []data;
    delete []result;
    delete []frontier_ids;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        //nec_test_performance();

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

        traversal_test(graph);

        BFS<int, float> bfs_operation(graph);

        #if defined(__USE_NEC_SX_AURORA__) || defined( __USE_INTEL__)
        int *bfs_result;
        bfs_operation.allocate_result_memory(graph.get_vertices_count(), &bfs_result);
        #endif

        #ifdef __USE_GPU__
        int *device_bfs_result;
        bfs_operation.allocate_device_result_memory(graph.get_vertices_count(), &device_bfs_result);
        #endif

        const int source_vertex_num = 6;
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
            vertex_to_check = i*200;//rand()%(non_zero_vertices - 1);
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
