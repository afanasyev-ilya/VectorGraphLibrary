#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ComputeOperation>
void __global__ compute_kernel_all_active(const int _frontier_size,
                                          //const long long *_vertex_pointers,
                                          ComputeOperation compute_op)
{
    const int src_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(src_id < _frontier_size)
    {
        int connections_count = 0;//_vertex_pointers[src_id + 1] - _vertex_pointers[src_id];
        int vector_index = lane_id();
        compute_op(src_id, connections_count, vector_index);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ComputeOperation>
void __global__ compute_kernel_dense(const int *_frontier_flags,
                                     const int _frontier_size,
                                     //const long long *_vertex_pointers,
                                     ComputeOperation compute_op)
{
    const int src_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(src_id < _frontier_size)
    {
        if(_frontier_flags[src_id] > 0)
        {
            int connections_count = 0;//_vertex_pointers[src_id + 1] - _vertex_pointers[src_id];
            int vector_index = lane_id();
            compute_op(src_id, connections_count, vector_index);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ComputeOperation>
void __global__ compute_kernel_sparse(const int *_frontier_ids,
                                      const int _frontier_size,
                                      //const long long *_vertex_pointers,
                                      ComputeOperation compute_op)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _frontier_size)
    {
        int src_id = _frontier_ids[idx];
        int connections_count = 0;//_vertex_pointers[src_id + 1] - _vertex_pointers[src_id];
        int vector_index = lane_id();
        compute_op(src_id, connections_count, vector_index);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ComputeOperation, typename GraphContainer, typename FrontierContainer>
void GraphAbstractionsGPU::compute_worker(GraphContainer &_graph,
                                          FrontierContainer &_frontier,
                                          ComputeOperation &&compute_op)
{
    int vertices_count = _graph.get_vertices_count();
    if(_frontier.get_sparsity_type() == ALL_ACTIVE_FRONTIER)
    {
        SAFE_KERNEL_CALL((compute_kernel_all_active <<< (vertices_count - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>>
                (vertices_count, /*vertex_pointers,*/ compute_op)));
    }
    else if(_frontier.get_sparsity_type() == DENSE_FRONTIER)
    {
        SAFE_KERNEL_CALL((compute_kernel_dense <<< (vertices_count - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>>
                (_frontier.flags, vertices_count, /*vertex_pointers,*/  compute_op)));
    }
    else if(_frontier.get_sparsity_type() == SPARSE_FRONTIER)
    {
        int frontier_size = _frontier.get_size();
        SAFE_KERNEL_CALL((compute_kernel_sparse <<< (frontier_size - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>>
                (_frontier.ids, frontier_size, /*vertex_pointers,*/  compute_op)));
    }
    cudaDeviceSynchronize();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ComputeOperation>
void GraphAbstractionsGPU::compute(VGL_Graph &_graph,
                                   VGL_Frontier &_frontier,
                                   ComputeOperation &&compute_op)
{
    this->common_compute(_graph, _frontier, compute_op, this);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

