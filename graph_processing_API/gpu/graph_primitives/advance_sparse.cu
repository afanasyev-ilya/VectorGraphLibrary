#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphPrimitivesGPU::advance_sparse(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                        FrontierGPU &_frontier,
                                        EdgeOperation edge_op,
                                        VertexPreprocessOperation vertex_preprocess_op,
                                        VertexPostprocessOperation vertex_postprocess_op)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    int grid_threshold_start = 0;
    int grid_threshold_end = 0;
    int block_threshold_start = 0;
    int block_threshold_end = 0;
    int warp_threshold_start = 0;
    int warp_threshold_end = 0;
    int thread_threshold_start = 0;
    int thread_threshold_end = 0;

    _frontier.split_sorted_frontier(outgoing_ptrs, grid_threshold_start, grid_threshold_end, block_threshold_start,
                                    block_threshold_end, warp_threshold_start, warp_threshold_end,
                                    thread_threshold_start, thread_threshold_end);

    int grid_vertices_count = grid_threshold_end - grid_threshold_start;
    if (grid_vertices_count > 0)
    {
        grid_per_vertex_kernel <<< grid_vertices_count, 1, 0, grid_processing_stream >>>
                                                              (outgoing_ptrs, outgoing_ids, _frontier.ids, vertices_count, grid_threshold_start,
                                                                      grid_threshold_end, edge_op, vertex_preprocess_op, vertex_postprocess_op);
    }

    int block_vertices_count = block_threshold_end - block_threshold_start;
    if (block_vertices_count > 0)
    {
        block_per_vertex_kernel <<< block_vertices_count, BLOCK_SIZE, 0, block_processing_stream >>>
                                                                         (outgoing_ptrs, outgoing_ids, _frontier.ids, vertices_count, block_threshold_start,
                                                                                 block_threshold_end, edge_op, vertex_preprocess_op, vertex_postprocess_op);
    }

    int warp_vertices_count = warp_threshold_end - warp_threshold_start;
    if (warp_vertices_count > 0)
    {
        warp_per_vertex_kernel <<< WARP_SIZE*(warp_vertices_count - 1)/BLOCK_SIZE + 1, BLOCK_SIZE, 0, warp_processing_stream >>>
                                                                                                      (outgoing_ptrs, outgoing_ids, _frontier.ids, vertices_count, warp_threshold_start,
                                                                                                              warp_threshold_end, edge_op, vertex_preprocess_op, vertex_postprocess_op);
    }

    int thread_vertices_count = thread_threshold_end - thread_threshold_start;
    if (thread_vertices_count > 0)
    {
        thread_per_vertex_kernel <<< (thread_vertices_count - 1) / BLOCK_SIZE + 1, BLOCK_SIZE, 0, thread_processing_stream >>>
                                                                                                  (outgoing_ptrs, outgoing_ids, _frontier.ids, vertices_count, thread_threshold_start,
                                                                                                          thread_threshold_end, edge_op, vertex_preprocess_op, vertex_postprocess_op);
    }
    cudaDeviceSynchronize();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class EdgeOperation>
void __global__ grid_per_vertex_kernel_child(const long long *_vertex_pointers,
                                             const int *_adjacent_ids,
                                             const int _vertices_count,
                                             const int _src_id,
                                             const int _connections_count,
                                             EdgeOperation edge_op)
{
    const int src_id = _src_id;
    const long long edge_start = _vertex_pointers[src_id];
    const long long edge_pos = blockIdx.x * blockDim.x + threadIdx.x;

    if(edge_pos < _connections_count)
    {
        const long long int global_edge_pos = edge_start + edge_pos;
        const int dst_id = _adjacent_ids[global_edge_pos];
        const int local_edge_pos = edge_pos;
        edge_op(src_id, dst_id, local_edge_pos, global_edge_pos);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class EdgeOperation, class VertexPreprocessOperation, class VertexPostprocessOperation>
void __global__ grid_per_vertex_kernel(const long long *_vertex_pointers,
                                       const int *_adjacent_ids,
                                       const int *_frontier_ids,
                                       const int _vertices_count,
                                       const int _vertex_part_start,
                                       const int _vertex_part_end,
                                       EdgeOperation edge_op,
                                       VertexPreprocessOperation vertex_preprocess_op,
                                       VertexPostprocessOperation vertex_postprocess_op)
{
    const int frontier_pos = blockIdx.x * blockDim.x + threadIdx.x + _vertex_part_start;
    if(frontier_pos < _vertex_part_end)
    {
        const int src_id = _frontier_ids[frontier_pos];
        const int connections_count = _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];

        vertex_preprocess_op(src_id, connections_count);

        dim3 child_threads(BLOCK_SIZE);
        dim3 child_blocks((connections_count - 1) / BLOCK_SIZE + 1);
        grid_per_vertex_kernel_child <<< child_blocks, child_threads >>> (_vertex_pointers, _adjacent_ids,
                _vertices_count, src_id, connections_count, edge_op);

        vertex_postprocess_op(src_id, connections_count);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class EdgeOperation, class VertexPreprocessOperation, class VertexPostprocessOperation>
void __global__ block_per_vertex_kernel(const long long *_vertex_pointers,
                                        const int *_adjacent_ids,
                                        const int *_frontier_ids,
                                        const int _vertices_count,
                                        const int _vertex_part_start,
                                        const int _vertex_part_end,
                                        EdgeOperation edge_op,
                                        VertexPreprocessOperation vertex_preprocess_op,
                                        VertexPostprocessOperation vertex_postprocess_op)
{
    const int frontier_pos = blockIdx.x + _vertex_part_start;
    if(frontier_pos < _vertex_part_end)
    {
        const int src_id = _frontier_ids[frontier_pos];
        const long long edge_start = _vertex_pointers[src_id];
        const int connections_count =  _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];
        vertex_preprocess_op(src_id, connections_count);

        for(register int edge_pos = threadIdx.x; edge_pos < connections_count; edge_pos += BLOCK_SIZE)
        {
            if(edge_pos < connections_count)
            {
                const long long int global_edge_pos = edge_start + edge_pos;
                const int dst_id = _adjacent_ids[global_edge_pos];
                const int local_edge_pos = edge_pos;
                edge_op(src_id, dst_id, local_edge_pos, global_edge_pos);
            }
        }

        vertex_postprocess_op(src_id, connections_count);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class EdgeOperation, class VertexPreprocessOperation, class VertexPostprocessOperation>
void __global__ warp_per_vertex_kernel(const long long *_vertex_pointers,
                                       const int *_adjacent_ids,
                                       const int *_frontier_ids,
                                       const int _vertices_count,
                                       const int _vertex_part_start,
                                       const int _vertex_part_end,
                                       EdgeOperation edge_op,
                                       VertexPreprocessOperation vertex_preprocess_op,
                                       VertexPostprocessOperation vertex_postprocess_op)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int frontier_pos = blockIdx.x * (blockDim.x/ WARP_SIZE) + warp_id + _vertex_part_start;

    if(frontier_pos < _vertex_part_end)
    {
        const int src_id = _frontier_ids[frontier_pos];
        const long long edge_start = _vertex_pointers[src_id];
        const int connections_count = _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];
        vertex_preprocess_op(src_id, connections_count);

        for(register int edge_pos = lane_id; edge_pos < connections_count; edge_pos += WARP_SIZE)
        {
            if(edge_pos < connections_count)
            {
                const long long int global_edge_pos = edge_start + edge_pos;
                const int dst_id = _adjacent_ids[global_edge_pos];
                const int local_edge_pos = edge_pos;
                edge_op(src_id, dst_id, local_edge_pos, global_edge_pos);
            }
        }

        vertex_postprocess_op(src_id, connections_count);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation, typename VertexPostprocessOperation>
void __global__ thread_per_vertex_kernel(const long long *_vertex_pointers,
                                         const int *_adjacent_ids,
                                         const int *_frontier_ids,
                                         const int _vertices_count,
                                         const int _vertex_part_start,
                                         const int _vertex_part_end,
                                         EdgeOperation edge_op,
                                         VertexPreprocessOperation vertex_preprocess_op,
                                         VertexPostprocessOperation vertex_postprocess_op)
{
    const int frontier_pos = blockIdx.x * blockDim.x + threadIdx.x + _vertex_part_start;

    if(frontier_pos < _vertex_part_end)
    {
        const int src_id = _frontier_ids[frontier_pos];

        const long long edge_start = _vertex_pointers[src_id];
        const int connections_count = _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];

        vertex_preprocess_op(src_id, connections_count);

        for(register int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            if(edge_pos < connections_count)
            {
                const long long int global_edge_pos = edge_start + edge_pos;
                const int dst_id = _adjacent_ids[global_edge_pos];
                const int local_edge_pos = edge_pos;
                edge_op(src_id, dst_id, local_edge_pos, global_edge_pos);
            }
        }

        vertex_postprocess_op(src_id, connections_count);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation, typename VertexPostprocessOperation>
void __global__ remaining_vertices_kernel(const long long *_vector_group_pointers,
                                          const int *_vector_group_sizes,
                                          const int _number_of_vertices_in_first_part,
                                          const int *_adjacent_ids,
                                          const int *_frontier_flags,
                                          const int _vertices_count,
                                          EdgeOperation edge_op,
                                          VertexPreprocessOperation vertex_preprocess_op,
                                          VertexPostprocessOperation vertex_postprocess_op)
{
    const int src_id = blockIdx.x * blockDim.x + threadIdx.x + _number_of_vertices_in_first_part;

    if(src_id < _vertices_count)
    {
        const int segment_connections_count  = _vector_group_sizes[blockIdx.x];
        if(segment_connections_count > 0)
        {
            if(_frontier_flags[src_id] == IN_FRONTIER_FLAG)
            {
                vertex_preprocess_op(src_id, segment_connections_count);

                const long long int segment_edges_start = _vector_group_pointers[blockIdx.x];
                for(register int edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
                {
                    const long long int global_edge_pos = segment_edges_start + edge_pos * blockDim.x + threadIdx.x;
                    const int dst_id = _adjacent_ids[global_edge_pos];
                    const int local_edge_pos = edge_pos;

                    edge_op(src_id, dst_id, local_edge_pos, global_edge_pos);
                }

                vertex_postprocess_op(src_id, segment_connections_count);
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////