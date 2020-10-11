#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void UndirectedCSRGraph::move_to_device()
{
    if(this->graph_on_device)
    {
        return;
    }
    
    this->graph_on_device = true;

    MemoryAPI::move_array_to_device<long long>(&vertex_pointers, this->vertices_count + 1);
    MemoryAPI::move_array_to_device<int>(&adjacent_ids, this->edges_count);

    #ifdef __USE_MANAGED_MEMORY__
    MemoryAPI::prefetch_managed_array(vertex_pointers, this->vertices_count + 1);
    MemoryAPI::prefetch_managed_array(adjacent_ids, this->edges_count);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void UndirectedCSRGraph::move_to_host()
{
    if(!this->graph_on_device)
    {
        return;
    }
    
    this->graph_on_device = false;

    MemoryAPI::move_array_to_host<long long>(&vertex_pointers, this->vertices_count + 1);
    MemoryAPI::move_array_to_host<int>(&adjacent_ids, this->edges_count);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void UndirectedCSRGraph::estimate_gpu_thresholds()
{
    gpu_grid_threshold_vertex = 0;
    gpu_block_threshold_vertex = 0;
    gpu_warp_threshold_vertex = 0;
    
    for(int i = 0; i < this->vertices_count - 1; i++)
    {
        int current_size = vertex_pointers[i+1] - vertex_pointers[i];
        int next_size = vertex_pointers[i+2] - vertex_pointers[i+1];
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

