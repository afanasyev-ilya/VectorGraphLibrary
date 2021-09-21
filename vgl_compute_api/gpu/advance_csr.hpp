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
                                                       VertexPostprocessOperation vertex_postprocess_op,
                                                       bool _sparse_mode)
{
    const register int frontier_pos = blockIdx.x;

    if(frontier_pos < _frontier_size)
    {
        int src_id = frontier_pos;
        if(_sparse_mode)
            src_id = _frontier_ids[frontier_pos];

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
                                               VertexPostprocessOperation vertex_postprocess_op,
                                               bool _sparse_mode)
{
    const int virtual_warp_id = threadIdx.x / VirtualWarpSize;
    const int position_in_virtual_warp = threadIdx.x % VirtualWarpSize;

    const int frontier_pos = blockIdx.x * (blockDim.x / VirtualWarpSize) + virtual_warp_id;

    if(frontier_pos < _frontier_size)
    {
        int src_id = frontier_pos;
        if(_sparse_mode)
            src_id = _frontier_ids[frontier_pos];

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

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation>
void GraphAbstractionsGPU::advance_worker(CSRGraph &_graph,
                                          FrontierCSR &_frontier,
                                          EdgeOperation &&edge_op,
                                          VertexPreprocessOperation &&vertex_preprocess_op,
                                          VertexPostprocessOperation &&vertex_postprocess_op,
                                          CollectiveEdgeOperation &&collective_edge_op,
                                          CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                                          CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op,
                                          bool _inner_mpi_processing)
{
    Timer tm;
    tm.start();
    LOAD_CSR_GRAPH_DATA(_graph);
    LOAD_FRONTIER_DATA(_frontier);

    long long process_shift = compute_process_shift(current_traversal_direction, CSR_STORAGE);

    if(_frontier.get_sparsity_type() == ALL_ACTIVE_FRONTIER)
    {
        dim3 grid((vertices_count - 1) / BLOCK_SIZE + 1);
        csr_all_active_advance_kernel<<<grid, BLOCK_SIZE>>>(vertex_pointers, adjacent_ids, vertices_count,
                process_shift, edge_op, vertex_preprocess_op,
                vertex_postprocess_op);
    }
    else if(_frontier.get_sparsity_type() == SPARSE_FRONTIER)
    {
        dim3 grid((frontier_size - 1) / BLOCK_SIZE + 1);
        csr_sparse_advance_kernel<<<grid, BLOCK_SIZE>>>(vertex_pointers, adjacent_ids, frontier_ids, frontier_size,
                process_shift, edge_op, vertex_preprocess_op,
                vertex_postprocess_op);
    }

    cudaDeviceSynchronize();

    size_t work = frontier_neighbours_count;

    tm.end();

    performance_stats.update_advance_stats(tm.get_time(), work*(INT_ELEMENTS_PER_EDGE)*sizeof(int), work);

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_and_bandwidth_stats("Advance (CSR)", work, (INT_ELEMENTS_PER_EDGE + 1)*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation>
void GraphAbstractionsGPU::advance_worker(CSR_VG_Graph &_graph,
                                          FrontierCSR_VG &_frontier,
                                          EdgeOperation &&edge_op,
                                          VertexPreprocessOperation &&vertex_preprocess_op,
                                          VertexPostprocessOperation &&vertex_postprocess_op,
                                          CollectiveEdgeOperation &&collective_edge_op,
                                          CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                                          CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op,
                                          bool _inner_mpi_processing)
{
    Timer tm;
    tm.start();
    LOAD_CSR_GRAPH_DATA(_graph);
    LOAD_FRONTIER_DATA(_frontier);

    long long process_shift = compute_process_shift(current_traversal_direction, CSR_STORAGE);

    if(_frontier.vertex_groups[0].get_size() > 0)
    {
        dim3 grid(_frontier.vertex_groups[0].get_size());
        dim3 block(BLOCK_SIZE);
        vg_csr_advance_block_per_vertex_kernel<<<grid, block, 0, stream_1>>>(vertex_pointers, adjacent_ids,
                                                      _frontier.vertex_groups[0].get_ids(), _frontier.vertex_groups[0].get_size(),
                                                      process_shift, edge_op, vertex_preprocess_op,
                                                      vertex_postprocess_op, true);
    }
    if(_frontier.vertex_groups[1].get_size() > 0)
    {
        dim3 grid((_frontier.vertex_groups[1].get_size() - 1) / (BLOCK_SIZE/32) + 1);
        dim3 block(BLOCK_SIZE);
        virtual_warp_per_vertex_kernel<32><<<grid, block, 0, stream_2>>>(vertex_pointers, adjacent_ids,
                                                      _frontier.vertex_groups[1].get_ids(), _frontier.vertex_groups[1].get_size(),
                                                      process_shift, edge_op, vertex_preprocess_op,
                                                      vertex_postprocess_op, true);
    }
    if(_frontier.vertex_groups[2].get_size() > 0)
    {
        dim3 grid((_frontier.vertex_groups[2].get_size() - 1) / (BLOCK_SIZE/16) + 1);
        dim3 block(BLOCK_SIZE);
        virtual_warp_per_vertex_kernel<16><<<grid, block, 0, stream_3>>>(vertex_pointers, adjacent_ids,
                                                      _frontier.vertex_groups[2].get_ids(), _frontier.vertex_groups[2].get_size(),
                                                      process_shift, edge_op, vertex_preprocess_op,
                                                      vertex_postprocess_op, true);
    }
    if(_frontier.vertex_groups[3].get_size() > 0)
    {
        dim3 grid((_frontier.vertex_groups[3].get_size() - 1) / (BLOCK_SIZE/8) + 1);
        dim3 block(BLOCK_SIZE);
        virtual_warp_per_vertex_kernel<8><<<grid, block, 0, stream_4>>>(vertex_pointers, adjacent_ids,
                                                      _frontier.vertex_groups[3].get_ids(), _frontier.vertex_groups[3].get_size(),
                                                      process_shift, edge_op, vertex_preprocess_op,
                                                      vertex_postprocess_op, true);
    }
    if(_frontier.vertex_groups[4].get_size() > 0)
    {
        dim3 grid((_frontier.vertex_groups[4].get_size() - 1) / (BLOCK_SIZE/4) + 1);
        dim3 block(BLOCK_SIZE);
        virtual_warp_per_vertex_kernel<4><<<grid, block, 0, stream_5>>>(vertex_pointers, adjacent_ids,
                                                      _frontier.vertex_groups[4].get_ids(), _frontier.vertex_groups[4].get_size(),
                                                      process_shift, edge_op, vertex_preprocess_op,
                                                      vertex_postprocess_op, true);
    }
    if(_frontier.vertex_groups[5].get_size() > 0)
    {
        dim3 grid((_frontier.vertex_groups[5].get_size() - 1) / (BLOCK_SIZE) + 1);
        dim3 block(BLOCK_SIZE);
        virtual_warp_per_vertex_kernel<1><<<grid, block, 0, stream_6>>>(vertex_pointers, adjacent_ids,
                                                      _frontier.vertex_groups[5].get_ids(), _frontier.vertex_groups[5].get_size(),
                                                      process_shift, edge_op, vertex_preprocess_op,
                                                      vertex_postprocess_op, true);
    }

    cudaDeviceSynchronize();

    size_t work = frontier_neighbours_count;

    tm.end();

    performance_stats.update_advance_stats(tm.get_time(), work*(INT_ELEMENTS_PER_EDGE)*sizeof(int), work);

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_and_bandwidth_stats("Advance (CSR)", work, (INT_ELEMENTS_PER_EDGE + 1)*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
