//
//  custom_test.cpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 04/10/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"
#include <iostream>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        double t1,t2;
        cout << "Custom test..." << endl;
        
        // parse args
        AlgorithmCommandOptionsParser parser;
        parser.parse_args(argc, argv);
        
        EdgesListGraph<int, float> rand_graph;
        int vertices_count = pow(2.0, atoi(argv[1]));
        int edges_count = vertices_count * atoi(argv[2]);
        GraphGenerationAPI<int, float>::random_uniform(rand_graph, vertices_count, edges_count, UNDIRECTED_GRAPH);
        
        ExtendedCSRGraph<int, float> graph;
        graph.import_graph(rand_graph, VERTICES_SORTED, EDGES_UNSORTED, VECTOR_LENGTH, PUSH_TRAVERSAL);
        
        long long *outgoing_ptrs    = graph.get_outgoing_ptrs   ();
        int       *outgoing_ids     = graph.get_outgoing_ids    ();
        float     *outgoing_weights = graph.get_outgoing_weights();
        
        int v1 = outgoing_ids[10];
        int v2 = outgoing_ids[11];
        uint64_t *packed_ids = (uint64_t*)outgoing_ids;
        uint64_t v_tmp = packed_ids[10/2];
        uint32_t v4 = (uint32_t)((v_tmp & 0xFFFFFFFF00000000LL) >> 32);
        uint32_t v3 = (uint32_t)(v_tmp & 0xFFFFFFFFLL);
        
        cout << v1 << " " << v2 << " " << v3 << " " << v4 << endl;
        
        int active_count = vertices_count / atoi(argv[3]);
        int *active_ids = new int[active_count];
        float *data = new float[vertices_count];
        int *levels = new int[vertices_count];
        
        for(int i = 0; i < vertices_count; i++)
        {
            levels[i] = rand()%4;
        }
        
        // get edges count
        long long real_edges_count = 0;
        for(int i = 0; i < vertices_count; i++)
        {
            int src_id = i;
            
            long long edge_start = outgoing_ptrs[src_id];
            int connections_count = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
            for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
            {
                real_edges_count++;
            }
        }

        // pack data
        uint64_t *packed_vals = new uint64_t[real_edges_count];
        //#pragma omp parallel for schedule(static)
        for(long long i = 0; i < real_edges_count; i++)
        {
            uint32_t dst_id = outgoing_ids[i];
            uint32_t weight = outgoing_weights[i] + (5);
            uint64_t tmp_val = ((uint64_t)dst_id ) << 32 | weight;
            packed_vals[i] = tmp_val;
        }
        
        int step = atoi(argv[3]) - 1;
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < active_count; i++)
        {
            active_ids[i] = i + step;
        }
        
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < vertices_count; i++)
        {
            data[i] = i;
        }
        
        double traversed_edges = 0;
        for(int i = 0; i < active_count; i++)
        {
            int src_id = active_ids[i];
            int connections_count = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
            traversed_edges += connections_count;
        }
        
        t1 = omp_get_wtime();
        for(int i = 0; i < active_count; i++)
        {
            int src_id = active_ids[i];
            
            long long edge_start = outgoing_ptrs[src_id];
            int connections_count = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
            for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
            {
                int dst_id = outgoing_ids[edge_start + edge_pos];
                float weight = outgoing_weights[edge_start + edge_pos];
                
                if(data[dst_id] < weight)
                {
                    data[src_id] += weight + dst_id;
                }
            }
        }
        t2 = omp_get_wtime();
        cout << "Trivial Band: " << (4.0 * sizeof(int) * traversed_edges) / ((t2 - t1)*1e9) << " GB/s" << endl;
        cout << "Trivial Perf: " << ((double)traversed_edges)/((t2 - t1)*1e6) << " MTEPS" << endl;
        
        int connections[VECTOR_LENGTH];
        int active_reg[VECTOR_LENGTH];
        long long start_pos[VECTOR_LENGTH];
        
        #pragma _NEC vreg(start_pos)
        #pragma _NEC vreg(connections)
        #pragma _NEC vreg(active_reg)
        
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            connections[i] = 0;
            active_ids[i] = 0;
            start_pos[i] = 0;
        }
        
        #pragma _NEC retain(data)
        
        t1 = omp_get_wtime();
        #pragma omp parallel for schedule(static, 1)
        for(int vec_start = 0; vec_start < active_count; vec_start += VECTOR_LENGTH)
        {
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                active_reg[i] = active_ids[vec_start + i];
                
                int src_id = active_reg[i];
                connections[i] = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
                start_pos[i] = outgoing_ptrs[src_id];
            }
            
            int max_connections = 0;
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                if(max_connections < connections[i])
                    max_connections = connections[i];
            }
            
            for(int edge_pos = 0; edge_pos < max_connections; edge_pos += 4)
            {
                #ifdef __USE_NEC_SX_AURORA__
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                #endif
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int idx = i;
                    int src_id = active_reg[idx];
                    int shift = edge_pos;
                    uint64_t tmp_val1 = packed_vals[start_pos[idx] + shift];
                    uint64_t tmp_val2 = packed_vals[start_pos[idx] + shift + 1];
                    uint64_t tmp_val3 = packed_vals[start_pos[idx] + shift + 1];
                    uint64_t tmp_val4 = packed_vals[start_pos[idx] + shift + 3];
                                       
                    if((shift) < connections[idx])
                    {
                        uint32_t dst_id = (uint32_t)((tmp_val1 & 0xFFFFFFFF00000000LL) >> 32);
                        uint32_t weight = (uint32_t)(tmp_val1 & 0xFFFFFFFFLL);
                                               
                        if(data[dst_id] < weight)
                        {
                            data[src_id] += weight + dst_id;
                        }
                    }
                    
                    if((shift + 1) < connections[idx])
                    {
                        uint32_t dst_id = (uint32_t)((tmp_val2 & 0xFFFFFFFF00000000LL) >> 32);
                        uint32_t weight = (uint32_t)(tmp_val2 & 0xFFFFFFFFLL);
                                               
                        if(data[dst_id] < weight)
                        {
                            data[src_id] += weight + dst_id;
                        }
                    }
                    
                    if((shift + 2) < connections[idx])
                    {
                        uint32_t dst_id = (uint32_t)((tmp_val3 & 0xFFFFFFFF00000000LL) >> 32);
                        uint32_t weight = (uint32_t)(tmp_val3 & 0xFFFFFFFFLL);
                                               
                        if(data[dst_id] < weight)
                        {
                            data[src_id] += weight + dst_id;
                        }
                    }
                    
                    if((shift + 3) < connections[idx])
                    {
                        uint32_t dst_id = (uint32_t)((tmp_val4 & 0xFFFFFFFF00000000LL) >> 32);
                        uint32_t weight = (uint32_t)(tmp_val4 & 0xFFFFFFFFLL);
                                               
                        if(data[dst_id] < weight)
                        {
                            data[src_id] += weight + dst_id;
                        }
                    }
                }
            }
        }
        t2 = omp_get_wtime();
        cout << "Weighted Band: " << (4.0 * sizeof(int) * traversed_edges) / ((t2 - t1)*1e9) << " GB/s" << endl;
        cout << "Weighted Perf: " << ((double)traversed_edges)/((t2 - t1)*1e6) << " MTEPS" << endl;
        
        int cur_level = 2;
        
        t1 = omp_get_wtime();
        #pragma omp parallel for schedule(static, 1)
        for(int vec_start = 0; vec_start < active_count; vec_start += VECTOR_LENGTH)
        {
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                active_reg[i] = active_ids[vec_start + i];
                
                int src_id = active_reg[i];
                connections[i] = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
                start_pos[i] = outgoing_ptrs[src_id];
            }
            
            int max_connections = 0;
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                if(max_connections < connections[i])
                    max_connections = connections[i];
            }
            
            for(int edge_pos = 0; edge_pos < max_connections; edge_pos += 2)
            {
                #ifdef __USE_NEC_SX_AURORA__
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                #endif
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int src_id = active_reg[i];
                    int dst_id = outgoing_ids[start_pos[i] + edge_pos];
                    int dst_id2 = outgoing_ids[start_pos[i] + edge_pos + 1];
                    
                    if((edge_pos < connections[i]) && (levels[dst_id] == cur_level))
                    {
                        levels[src_id] = cur_level + 1;
                    }
                    
                    if(((edge_pos + 1) < connections[i]) && (levels[dst_id] == cur_level))
                    {
                        levels[src_id] = cur_level + 1;
                    }
                }
            }
        }
        t2 = omp_get_wtime();
        cout << "Edges vect Band: " << (4.0 * sizeof(int) * traversed_edges) / ((t2 - t1)*1e9) << " GB/s" << endl;
        cout << "Edges vect Perf: " << ((double)traversed_edges)/((t2 - t1)*1e6) << " MTEPS" << endl;
        
        for(int i = 0; i < vertices_count; i++)
        {
            levels[i] = rand()%4;
        }
        
        t1 = omp_get_wtime();
        #pragma omp parallel for schedule(static, 1)
        for(int vec_start = 0; vec_start < active_count; vec_start += VECTOR_LENGTH)
        {
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                active_reg[i] = active_ids[vec_start + i];
                
                int src_id = active_reg[i];
                connections[i] = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
                start_pos[i] = outgoing_ptrs[src_id];
            }
            
            int max_connections = 0;
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                if(max_connections < connections[i])
                    max_connections = connections[i];
            }
            
            for(int edge_pos = 0; edge_pos < max_connections; edge_pos += 2)
            {
                #ifdef __USE_NEC_SX_AURORA__
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                #endif
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int src_id = active_reg[i];
                    int dst_id = outgoing_ids[start_pos[i] + edge_pos];
                    int dst_id2 = outgoing_ids[start_pos[i] + edge_pos + 1];
                    
                    if((edge_pos < connections[i]) && (levels[src_id] == cur_level))
                    {
                        levels[dst_id] = cur_level + 1;
                    }
                    
                    if(((edge_pos + 1) < connections[i]) && (levels[src_id] == cur_level))
                    {
                        levels[dst_id] = cur_level + 1;
                    }
                }
            }
        }
        t2 = omp_get_wtime();
        cout << "scatter Band: " << (4.0 * sizeof(int) * traversed_edges) / ((t2 - t1)*1e9) << " GB/s" << endl;
        cout << "scatter Perf: " << ((double)traversed_edges)/((t2 - t1)*1e6) << " MTEPS" << endl;
        
        delete []active_ids;
        delete []data;
        delete []packed_vals;
        delete []levels;
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

