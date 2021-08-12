#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class EdgeOperation>
void __global__ edges_list_advance_kernel(int *_src_ids,
                                          int *_dst_ids,
                                          long long _edges_count,
                                          EdgeOperation edge_op)
{
    const register size_t edge_pos = blockIdx.x * blockDim.x + threadIdx.x;
    if(edge_pos < _edges_count)
    {
        const int src_id = _src_ids[edge_pos];
        const int dst_id = _dst_ids[edge_pos];
        int vector_index = lane_id();
        long long edge_pos = edge_pos;
        edge_op(src_id, dst_id, edge_pos, edge_pos, vector_index);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation, typename GraphContainer, typename FrontierContainer>
void GraphAbstractionsGPU::advance_worker(GraphContainer &_graph,
                                          FrontierContainer &_frontier,
                                          EdgeOperation &&edge_op,
                                          VertexPreprocessOperation &&vertex_preprocess_op,
                                          VertexPostprocessOperation &&vertex_postprocess_op,
                                          CollectiveEdgeOperation &&collective_edge_op,
                                          CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                                          CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op,
                                          bool _inner_mpi_processing)
{
    throw "Error in GraphAbstractionsGPU::advance : not implemented yet";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation>
void GraphAbstractionsGPU::advance_worker(EdgesListGraph &_graph,
                                          FrontierGeneral &_frontier,
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
    LOAD_EDGES_LIST_GRAPH_DATA(_graph);

    SAFE_KERNEL_CALL(( edges_list_advance_kernel<<< (edges_count - 1)/BLOCK_SIZE + 1, BLOCK_SIZE >>>(src_ids, dst_ids, edges_count, edge_op) ));

    tm.end();

    long long work = edges_count;
    performance_stats.update_advance_stats(tm.get_time(), work*(INT_ELEMENTS_PER_EDGE + 1)*sizeof(int), work);

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_and_bandwidth_stats("Advance (edges list)", work, (INT_ELEMENTS_PER_EDGE + 1)*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation>
void GraphAbstractionsGPU::advance_worker(CSRGraph &_graph,
                                          FrontierGeneral &_frontier,
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

    #ifdef __USE_CSR_VERTEX_GROUPS__
    if(_frontier.large_degree.size > 0)
    {
        dim3 grid(_frontier.large_degree.size);
        dim3 block(BLOCK_SIZE);
        vg_csr_advance_block_per_vertex_kernel<<<grid, block, 0, stream_1>>>(vertex_pointers, adjacent_ids,
                                                      _frontier.large_degree.ids, _frontier.large_degree.size,
                                                      process_shift, edge_op, vertex_preprocess_op,
                                                      vertex_postprocess_op);
    }
    if(_frontier.degree_32_1024.size > 0)
    {
        dim3 grid((_frontier.degree_32_1024.size - 1) / (BLOCK_SIZE/32) + 1);
        dim3 block(BLOCK_SIZE);
        virtual_warp_per_vertex_kernel<32><<<grid, block, 0, stream_2>>>(vertex_pointers, adjacent_ids,
                                                      _frontier.degree_32_1024.ids, _frontier.degree_32_1024.size,
                                                      process_shift, edge_op, vertex_preprocess_op,
                                                      vertex_postprocess_op);
    }
    if(_frontier.degree_16_32.size > 0)
    {
        dim3 grid((_frontier.degree_16_32.size - 1) / (BLOCK_SIZE/16) + 1);
        dim3 block(BLOCK_SIZE);
        virtual_warp_per_vertex_kernel<16><<<grid, block, 0, stream_3>>>(vertex_pointers, adjacent_ids,
                                                      _frontier.degree_16_32.ids, _frontier.degree_16_32.size,
                                                      process_shift, edge_op, vertex_preprocess_op,
                                                      vertex_postprocess_op);
    }
    if(_frontier.degree_8_16.size > 0)
    {
        dim3 grid((_frontier.degree_8_16.size - 1) / (BLOCK_SIZE/8) + 1);
        dim3 block(BLOCK_SIZE);
        virtual_warp_per_vertex_kernel<8><<<grid, block, 0, stream_4>>>(vertex_pointers, adjacent_ids,
                                                      _frontier.degree_8_16.ids, _frontier.degree_8_16.size,
                                                      process_shift, edge_op, vertex_preprocess_op,
                                                      vertex_postprocess_op);
    }
    if(_frontier.degree_4_8.size > 0)
    {
        dim3 grid((_frontier.degree_4_8.size - 1) / (BLOCK_SIZE/4) + 1);
        dim3 block(BLOCK_SIZE);
        virtual_warp_per_vertex_kernel<4><<<grid, block, 0, stream_5>>>(vertex_pointers, adjacent_ids,
                                                      _frontier.degree_4_8.ids, _frontier.degree_4_8.size,
                                                      process_shift, edge_op, vertex_preprocess_op,
                                                      vertex_postprocess_op);
    }
    if(_frontier.degree_0_4.size > 0)
    {
        dim3 grid((_frontier.degree_0_4.size - 1) / (BLOCK_SIZE) + 1);
        dim3 block(BLOCK_SIZE);
        virtual_warp_per_vertex_kernel<1><<<grid, block, 0, stream_6>>>(vertex_pointers, adjacent_ids,
                                                      _frontier.degree_0_4.ids, _frontier.degree_0_4.size,
                                                      process_shift, edge_op, vertex_preprocess_op,
                                                      vertex_postprocess_op);
    }
    size_t work = _frontier.large_degree.neighbours + _frontier.degree_32_1024.neighbours +
             _frontier.degree_16_32.neighbours + _frontier.degree_8_16.neighbours +
             _frontier.degree_4_8.neighbours + _frontier.degree_0_4.neighbours;
    #else
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
    size_t work = frontier_neighbours_count;
    #endif
    cudaDeviceSynchronize();

    tm.end();

    performance_stats.update_advance_stats(tm.get_time(), work*(INT_ELEMENTS_PER_EDGE)*sizeof(int), work);

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_and_bandwidth_stats("Advance (CSR)", work, (INT_ELEMENTS_PER_EDGE + 1)*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
