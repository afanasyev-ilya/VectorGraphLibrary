/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void __global__ csr_sparse_advance_kernel(long long *_vertex_pointers,
                                          int *_adjacent_ids,
                                          int *_frontier_ids,
                                          int _frontier_size,
                                          long long _process_shift,
                                          EdgeOperation edge_op,
                                          VertexPreprocessOperation vertex_preprocess_op,
                                          VertexPostprocessOperation vertex_postprocess_op)
{
    const register int front_pos = blockIdx.x * blockDim.x + threadIdx.x;
    if(front_pos < _frontier_size)
    {
        const int src_id = _frontier_ids[front_pos];

        const long long int start = _vertex_pointers[src_id];
        const long long int end = _vertex_pointers[src_id + 1];
        const int connections_count = end - start;

        vertex_preprocess_op(src_id, connections_count, 0);

        for (int local_edge_pos = 0; local_edge_pos < connections_count; local_edge_pos++)
        {
            const long long internal_edge_pos = start + local_edge_pos;
            const int vector_index = lane_id();
            const int dst_id = _adjacent_ids[internal_edge_pos];
            const long long external_edge_pos = _process_shift + internal_edge_pos;

            edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index);
        }

        vertex_postprocess_op(src_id, connections_count, 0);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void __global__ csr_all_active_advance_kernel(long long *_vertex_pointers,
                                              int *_adjacent_ids,
                                              int _vertices_count,
                                              long long _process_shift,
                                              EdgeOperation edge_op,
                                              VertexPreprocessOperation vertex_preprocess_op,
                                              VertexPostprocessOperation vertex_postprocess_op)
{
    const register int src_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(src_id < _vertices_count)
    {
        const long long int start = _vertex_pointers[src_id];
        const long long int end = _vertex_pointers[src_id + 1];
        const int connections_count = end - start;

        vertex_preprocess_op(src_id, connections_count, 0);

        for (int local_edge_pos = 0; local_edge_pos < connections_count; local_edge_pos++)
        {
            const long long internal_edge_pos = start + local_edge_pos;
            const int vector_index = lane_id();
            const int dst_id = _adjacent_ids[internal_edge_pos];
            const long long external_edge_pos = _process_shift + internal_edge_pos;

            edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index);
        }

        vertex_postprocess_op(src_id, connections_count, 0);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void __global__ vg_csr_advance_block_per_vertex_kernel(long long *_vertex_pointers,
                                                       int *_adjacent_ids,
                                                       int *_frontier_ids,
                                                       int _frontier_size,
                                                       size_t _process_shift,
                                                       EdgeOperation edge_op,
                                                       VertexPreprocessOperation vertex_preprocess_op,
                                                       VertexPostprocessOperation vertex_postprocess_op)
{
    const register int frontier_pos = blockIdx.x;

    if(frontier_pos < _frontier_size)
    {
        int src_id = _frontier_ids[frontier_pos];
        const register int tid = threadIdx.x;

        const long long int start = _vertex_pointers[src_id];
        const long long int end = _vertex_pointers[src_id + 1];
        const int connections_count = end - start;

        if(tid == 0)
            vertex_preprocess_op(src_id, connections_count, 0);

        for(int local_edge_pos = tid; local_edge_pos < connections_count; local_edge_pos += blockDim.x)
        {
            const long long internal_edge_pos = start + local_edge_pos;
            const int vector_index = lane_id();
            const int dst_id = _adjacent_ids[internal_edge_pos];
            const long long external_edge_pos = internal_edge_pos + _process_shift;

            edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index);
        }

        if(tid == 0)
            vertex_postprocess_op(src_id, connections_count, 0);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <int VirtualWarpSize, typename EdgeOperation, typename VertexPreprocessOperation, typename VertexPostprocessOperation>
void __global__ virtual_warp_per_vertex_kernel(const long long *_vertex_pointers,
                                               const int *_adjacent_ids,
                                               const int *_frontier_ids,
                                               const int _frontier_size,
                                               size_t _process_shift,
                                               EdgeOperation edge_op,
                                               VertexPreprocessOperation vertex_preprocess_op,
                                               VertexPostprocessOperation vertex_postprocess_op)
{
    const int virtual_warp_id = threadIdx.x / VirtualWarpSize;
    const int position_in_virtual_warp = threadIdx.x % VirtualWarpSize;

    const int frontier_pos = blockIdx.x * (blockDim.x / VirtualWarpSize) + virtual_warp_id;

    if(frontier_pos < _frontier_size)
    {
        const int src_id = _frontier_ids[frontier_pos];

        const long long start = _vertex_pointers[src_id];
        const long long end = _vertex_pointers[src_id + 1];
        const int connections_count = end - start;

        if(position_in_virtual_warp == 0)
            vertex_preprocess_op(src_id, frontier_pos, connections_count);

        for(int local_edge_pos = position_in_virtual_warp; local_edge_pos < connections_count; local_edge_pos += VirtualWarpSize)
        {
            const long long internal_edge_pos = start + local_edge_pos;
            const int vector_index = lane_id();
            const int dst_id = _adjacent_ids[internal_edge_pos];
            const long long external_edge_pos = internal_edge_pos + _process_shift;

            edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index);
        }

        if(position_in_virtual_warp == 0)
            vertex_postprocess_op(src_id, frontier_pos, connections_count);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
