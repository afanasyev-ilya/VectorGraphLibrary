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
        dim3 grid(frontier_part_size);
        cout << _frontier.get_vector_engine_part_size() << endl;
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
                    vertex_postprocess_op, true)));
        }
        else if (_frontier.get_vector_core_part_sparsity_type() == SPARSE_FRONTIER)
        {
            SAFE_KERNEL_CALL((virtual_warp_per_vertex_kernel<WARP_SIZE><<<grid, block, 0, stream_2>>>(vertex_pointers, adjacent_ids,
                    frontier_ids + shift, frontier_part_size, process_shift, edge_op, vertex_preprocess_op,
                    vertex_postprocess_op, false)));
        }
    }
    if(_frontier.get_collective_part_size() > 0)
    {
        int shift = _frontier.get_vector_engine_part_size() + _frontier.get_vector_core_part_size();
        int frontier_part_size = _frontier.get_collective_part_size();
        dim3 grid((frontier_part_size - 1) / (BLOCK_SIZE) + 1);
        if (_frontier.get_collective_part_sparsity_type() == ALL_ACTIVE_FRONTIER)
        {
            SAFE_KERNEL_CALL((virtual_warp_per_vertex_kernel<1><<<grid, block, 0, stream_2>>>(vertex_pointers, adjacent_ids,
                    frontier_ids + shift, frontier_part_size, process_shift, edge_op, vertex_preprocess_op,
                    vertex_postprocess_op, true)));
        }
        else if (_frontier.get_collective_part_sparsity_type() == SPARSE_FRONTIER)
        {
            SAFE_KERNEL_CALL((virtual_warp_per_vertex_kernel<1><<<grid, block, 0, stream_2>>>(vertex_pointers, adjacent_ids,
                    frontier_ids + shift, frontier_part_size, process_shift, edge_op, vertex_preprocess_op,
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
