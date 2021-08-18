/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation, typename VertexPostprocessOperation>
void __global__ vector_extension_advance_kernel(const long long *_vertex_pointers,
                                                const long long *_ve_vector_group_ptrs,
                                                const int *_ve_vector_group_sizes,
                                                const int *_ve_adjacent_ids,
                                                int _ve_starting_vertex,
                                                size_t _vertices_count,
                                                long long _process_shift,
                                                EdgeOperation edge_op,
                                                VertexPreprocessOperation vertex_preprocess_op,
                                                VertexPostprocessOperation vertex_postprocess_op)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int position_in_warp = threadIdx.x % WARP_SIZE;

    int idx = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id;
    int src_id = _ve_starting_vertex + src_id;

    if(src_id < _vertices_count)
    {
        int cur_vector_segment = idx / WARP_SIZE;
        int segment_first_vertex = cur_vector_segment * WARP_SIZE + _ve_starting_vertex;

        long long segment_edges_start = _ve_vector_group_ptrs[cur_vector_segment];
        int segment_connections_count = _ve_vector_group_sizes[cur_vector_segment];

        int connections_count = _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];

        vertex_preprocess_op(src_id, connections_count, position_in_warp);

        for(int edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
        {
            const int vector_index = position_in_warp;
            const long long internal_edge_pos = segment_edges_start + edge_pos * WARP_SIZE + position_in_warp;
            const int local_edge_pos = edge_pos;
            const long long external_edge_pos = _process_shift + internal_edge_pos;

            if(edge_pos < connections_count)
            {
                const int dst_id = _ve_adjacent_ids[internal_edge_pos];
                edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index);
            }
        }

        vertex_preprocess_op(src_id, connections_count, position_in_warp);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation>
void GraphAbstractionsGPU::advance_worker(VectorCSRGraph &_graph,
                                          FrontierVectorCSR &_frontier,
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
    LOAD_VECTOR_CSR_GRAPH_DATA(_graph);
    LOAD_FRONTIER_DATA(_frontier);

    long long process_shift = compute_process_shift(current_traversal_direction, CSR_STORAGE);

    dim3 block(BLOCK_SIZE);
    if(_frontier.get_vector_engine_part_size() > 0)
    {
        int frontier_part_size = _frontier.get_vector_engine_part_size();
        int shift = 0;
        dim3 grid(frontier_part_size);
        if(_frontier.get_vector_engine_part_sparsity_type() == ALL_ACTIVE_FRONTIER)
        {
            SAFE_KERNEL_CALL((vg_csr_advance_block_per_vertex_kernel<<<grid, block, 0, stream_1>>>(vertex_pointers, adjacent_ids,
                    frontier_ids, frontier_part_size, process_shift, edge_op, vertex_preprocess_op,
                    vertex_postprocess_op, false)));
        }
        else if (_frontier.get_vector_engine_part_sparsity_type() == SPARSE_FRONTIER)
        {
            SAFE_KERNEL_CALL((vg_csr_advance_block_per_vertex_kernel<<<grid, block, 0, stream_1>>>(vertex_pointers, adjacent_ids,
                    frontier_ids, frontier_part_size, process_shift, edge_op, vertex_preprocess_op, vertex_postprocess_op, true)));
        }
    }
    if(_frontier.get_vector_core_part_size() > 0)
    {
        int shift = _frontier.get_vector_engine_part_size();
        int frontier_part_size = _frontier.get_vector_core_part_size();
        dim3 grid((frontier_part_size - 1) / (BLOCK_SIZE/WARP_SIZE) + 1);
        if (_frontier.get_vector_core_part_sparsity_type() == ALL_ACTIVE_FRONTIER)
        {
            SAFE_KERNEL_CALL((virtual_warp_per_vertex_kernel<WARP_SIZE><<<grid, block, 0, stream_2>>>(vertex_pointers, adjacent_ids,
                    frontier_ids + shift, frontier_part_size, process_shift, edge_op, vertex_preprocess_op,
                    vertex_postprocess_op, false)));
        }
        else if (_frontier.get_vector_core_part_sparsity_type() == SPARSE_FRONTIER)
        {
            SAFE_KERNEL_CALL((virtual_warp_per_vertex_kernel<WARP_SIZE><<<grid, block, 0, stream_2>>>(vertex_pointers, adjacent_ids,
                    frontier_ids + shift, frontier_part_size, process_shift, edge_op, vertex_preprocess_op,
                    vertex_postprocess_op, true)));
        }
    }
    if(_frontier.get_collective_part_size() > 0)
    {
        int shift = _frontier.get_vector_engine_part_size() + _frontier.get_vector_core_part_size();
        int frontier_part_size = _frontier.get_collective_part_size();
        dim3 grid((frontier_part_size - 1) / (BLOCK_SIZE) + 1);
        if (_frontier.get_collective_part_sparsity_type() == ALL_ACTIVE_FRONTIER)
        {
            /*SAFE_KERNEL_CALL((virtual_warp_per_vertex_kernel<1><<<grid, block, 0, stream_3>>>(vertex_pointers, adjacent_ids,
                    frontier_ids, frontier_part_size, process_shift, edge_op, vertex_preprocess_op,
                    vertex_postprocess_op, false)));*/
            SAFE_KERNEL_CALL((vector_extension_advance_kernel<<<grid, block, 0, stream_3>>>(vertex_pointers,
                    ve_vector_group_ptrs, ve_vector_group_sizes, ve_adjacent_ids,
                    ve_starting_vertex, vertices_count, process_shift, edge_op, vertex_preprocess_op,
                    vertex_postprocess_op)));
        }
        else if (_frontier.get_collective_part_sparsity_type() == SPARSE_FRONTIER)
        {
            SAFE_KERNEL_CALL((virtual_warp_per_vertex_kernel<1><<<grid, block, 0, stream_3>>>(vertex_pointers, adjacent_ids,
                    frontier_ids, frontier_part_size, process_shift, edge_op, vertex_preprocess_op,
                    vertex_postprocess_op, true)));
        }
    }

    cudaDeviceSynchronize();

    size_t work = frontier_neighbours_count;

    tm.end();

    performance_stats.update_advance_stats(tm.get_time(), work*(INT_ELEMENTS_PER_EDGE)*sizeof(int), work);

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_and_bandwidth_stats("Advance (VectorCSR)", work, (INT_ELEMENTS_PER_EDGE + 1)*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
