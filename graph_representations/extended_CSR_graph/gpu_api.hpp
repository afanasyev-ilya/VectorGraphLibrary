//
//  gpu_api.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 17/01/2020.
//  Copyright © 2020 MSU. All rights reserved.
//

#ifndef gpu_api_hpp
#define gpu_api_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::move_to_device()
{
    if(this->graph_on_device)
    {
        return;
    }
    
    this->graph_on_device = true;
    
    move_array_to_device<_TVertexValue>(&(this->vertex_values), this->vertices_count);
    move_array_to_device<int>(&reordered_vertex_ids, this->vertices_count);
    move_array_to_device<long long>(&outgoing_ptrs, this->vertices_count + 1);
    move_array_to_device<int>(&outgoing_ids, this->edges_count);
    
    #ifdef __USE_WEIGHTED_GRAPHS__
    move_array_to_device<_TEdgeWeight>(&outgoing_weights, this->edges_count);
    #endif
    
    move_array_to_device<int>(&(vectorised_outgoing_ids), this->vertices_count*VECTOR_EXTENSION_SIZE);
    
    #ifdef __USE_WEIGHTED_GRAPHS__
    double device_memory_size = this->vertices_count*sizeof(int) + this->vertices_count* sizeof(int) + (this->vertices_count + 1)*sizeof(long long) + sizeof(int)*this->edges_count + sizeof(_TEdgeWeight)*this->edges_count + sizeof(int)*this->vertices_count*VECTOR_EXTENSION_SIZE;
    //cout << "allocated: " << device_memory_size/1e9 << " GB on device" << endl;
    #else
    double device_memory_size_no_weight = this->vertices_count*sizeof(int) + this->vertices_count* sizeof(int) + (this->vertices_count + 1)*sizeof(long long) + sizeof(int)*this->edges_count + sizeof(int)*this->vertices_count*5;
    //cout << "allocated without weights: " << device_memory_size_no_weight/1e9 << " GB on device" << endl;
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::move_to_host()
{
    if(!this->graph_on_device)
    {
        return;
    }
    
    this->graph_on_device = false;
    
    move_array_to_host<_TVertexValue>(&(this->vertex_values), this->vertices_count);
    move_array_to_host<int>(&reordered_vertex_ids, this->vertices_count);
    move_array_to_host<long long>(&outgoing_ptrs, this->vertices_count + 1);
    move_array_to_host<int>(&outgoing_ids, this->edges_count);
    
    #ifdef __USE_WEIGHTED_GRAPHS__
    move_array_to_host<_TEdgeWeight>(&outgoing_weights, this->edges_count);
    #endif
    
    move_array_to_host<int>(&(vectorised_outgoing_ids), this->vertices_count*VECTOR_EXTENSION_SIZE);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::estimate_gpu_thresholds()
{
    gpu_grid_threshold_vertex = 0;
    gpu_block_threshold_vertex = 0;
    gpu_warp_threshold_vertex = 0;
    
    for(int i = 0; i < this->vertices_count - 1; i++)
    {
        int current_size = outgoing_ptrs[i+1] - outgoing_ptrs[i];
        int next_size = outgoing_ptrs[i+2] - outgoing_ptrs[i+1];
        if((current_size > GPU_GRID_THRESHOLD_VALUE) && (next_size <= GPU_GRID_THRESHOLD_VALUE))
        {
            gpu_grid_threshold_vertex = i + 1;
        }
        if((current_size > GPU_BLOCK_THRESHOLD_VALUE) && (next_size <= GPU_BLOCK_THRESHOLD_VALUE))
        {
            gpu_block_threshold_vertex = i + 1;
        }
        if((current_size > GPU_WARP_THRESHOLD_VALUE) && (next_size <= GPU_WARP_THRESHOLD_VALUE))
        {
            gpu_warp_threshold_vertex = i + 1;
        }
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* gpu_api_h */
