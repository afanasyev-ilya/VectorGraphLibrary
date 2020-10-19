#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ set_all_active_frontier_kernel(int *_frontier_ids, int *_frontier_flags, int _vertices_count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < _vertices_count)
    {
        _frontier_ids[idx] = idx;
        _frontier_flags[idx] = IN_FRONTIER_FLAG;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierGPU::set_all_active()
{
    type = ALL_ACTIVE_FRONTIER;

    SAFE_KERNEL_CALL((set_all_active_frontier_kernel<<< (max_size - 1)/BLOCK_SIZE + 1, BLOCK_SIZE >>> (ids, flags, max_size)));
    current_size = max_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierGPU::add_vertex(int src_id)
{
    if(current_size > 0)
    {
        throw "Error in FrontierGPU::add_vertex: VGL can not add vertex to non-empty frontier";
    }
    ids[0] = src_id;
    flags[src_id] = IN_FRONTIER_FLAG;
    current_size = 1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierGPU::add_group_of_vertices(int *_vertex_ids, int _number_of_vertices)
{
    throw "Error FrontierGPU::add_group_of_vertices : not implemented yet";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
