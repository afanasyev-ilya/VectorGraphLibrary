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

template <typename ComputeOperation, typename Graph_Container>
void GraphAbstractionsGPU::compute_worker(Graph_Container &_graph,
                                          FrontierGeneral &_frontier,
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
void GraphAbstractionsGPU::compute_container_call(VGL_Graph &_graph,
                                                  VGL_Frontier &_frontier,
                                                  ComputeOperation &&compute_op)
{
    if(_graph.get_container_type() == CSR_GRAPH)
    {
        CSRGraph *container_graph = (CSRGraph *)_graph.get_direction_data(current_traversal_direction);
        FrontierGeneral *container_frontier = (FrontierGeneral *)_frontier.get_container_data();
        compute_worker(*container_graph, *container_frontier, compute_op);
    }
    else if(_graph.get_container_type() == EDGES_LIST_GRAPH)
    {
        EdgesListGraph *container_graph = (EdgesListGraph *)_graph.get_direction_data(current_traversal_direction);
        FrontierGeneral *container_frontier = (FrontierGeneral *)_frontier.get_container_data();
        compute_worker(*container_graph, *container_frontier, compute_op);
    }
    else
    {
        throw "Error in GraphAbstractionsGPU::compute : unsupported container type";
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ComputeOperation>
void GraphAbstractionsGPU::compute(VGL_Graph &_graph,
                                   VGL_Frontier &_frontier,
                                   ComputeOperation &&compute_op)
{
    Timer tm;
    tm.start();

    if(_frontier.get_direction() != current_traversal_direction) // TODO check
    {
        throw "Error in GraphAbstractionsGPU::compute : wrong frontier direction";
    }

    compute_container_call(_graph, _frontier, compute_op);

    tm.end();
    long long work = _frontier.size();
    performance_stats.update_compute_time(tm);
    performance_stats.update_bytes_requested(COMPUTE_INT_ELEMENTS*sizeof(int)*work);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("Compute", work, COMPUTE_INT_ELEMENTS*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

