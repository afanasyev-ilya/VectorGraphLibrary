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
            const long long external_edge_pos = internal_edge_pos;

            edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index);
        }

        vertex_postprocess_op(src_id, connections_count, 0);
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

    tm.end();

    long long work = frontier_neighbours_count;
    performance_stats.update_advance_stats(tm.get_time(), work*(INT_ELEMENTS_PER_EDGE)*sizeof(int), work);

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_and_bandwidth_stats("Advance (CSR)", work, (INT_ELEMENTS_PER_EDGE + 1)*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
