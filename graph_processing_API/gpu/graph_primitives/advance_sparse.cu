#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class EdgeOperation>
void __global__ grid_per_vertex_kernel_child(const long long *_vertex_pointers,
                                             const int *_adjacent_ids,
                                             const int _vertices_count,
                                             const int _src_id,
                                             const int _connections_count,
                                             EdgeOperation edge_op,
                                             int _frontier_pos,
                                             int *_new_frontier_ids,
                                             bool _generate_frontier,
                                             int *_new_frontier_size)
{
    const int src_id = _src_id;
    const long long edge_start = _vertex_pointers[src_id];
    const long long edge_pos = blockIdx.x * blockDim.x + threadIdx.x;

    if(edge_pos < _connections_count)
    {
        const long long int global_edge_pos = edge_start + edge_pos;
        const int dst_id = _adjacent_ids[global_edge_pos];
        const int local_edge_pos = edge_pos;
        edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, _frontier_pos);
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
                                       VertexPostprocessOperation vertex_postprocess_op,
                                       int *_new_frontier_ids,
                                       bool _generate_frontier,
                                       int *_new_frontier_size)
{
    const int frontier_pos = blockIdx.x * blockDim.x + threadIdx.x + _vertex_part_start;
    if(frontier_pos < _vertex_part_end)
    {
        const int src_id = _frontier_ids[frontier_pos];
        const int connections_count = _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];

        vertex_preprocess_op(src_id, frontier_pos, connections_count);

        dim3 child_threads(BLOCK_SIZE);
        dim3 child_blocks((connections_count - 1) / BLOCK_SIZE + 1);
        grid_per_vertex_kernel_child <<< child_blocks, child_threads >>> (_vertex_pointers, _adjacent_ids,
                _vertices_count, src_id, connections_count, edge_op, frontier_pos, _new_frontier_ids, _generate_frontier, _new_frontier_size);

        vertex_postprocess_op(src_id, frontier_pos, connections_count);
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
                                        VertexPostprocessOperation vertex_postprocess_op,
                                        int *_new_frontier_ids,
                                        bool _generate_frontier,
                                        int *_new_frontier_size)
{
    const int frontier_pos = blockIdx.x + _vertex_part_start;
    if(frontier_pos < _vertex_part_end)
    {
        const int src_id = _frontier_ids[frontier_pos];
        const long long edge_start = _vertex_pointers[src_id];
        const int connections_count =  _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];
        vertex_preprocess_op(src_id, frontier_pos, connections_count);

        for(register int edge_pos = threadIdx.x; edge_pos < connections_count; edge_pos += BLOCK_SIZE)
        {
            if(edge_pos < connections_count)
            {
                const long long int global_edge_pos = edge_start + edge_pos;
                const int dst_id = _adjacent_ids[global_edge_pos];
                const int local_edge_pos = edge_pos;
                edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, frontier_pos);
            }
        }

        vertex_postprocess_op(src_id, frontier_pos, connections_count);
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
                                       VertexPostprocessOperation vertex_postprocess_op,
                                       int *_new_frontier_ids,
                                       bool _generate_frontier,
                                       int *_new_frontier_size)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int frontier_pos = blockIdx.x * (blockDim.x/ WARP_SIZE) + warp_id + _vertex_part_start;

    if(frontier_pos < _vertex_part_end)
    {
        const int src_id = _frontier_ids[frontier_pos];
        const long long edge_start = _vertex_pointers[src_id];
        const int connections_count = _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];
        vertex_preprocess_op(src_id, frontier_pos, connections_count);

        for(register int edge_pos = lane_id; edge_pos < connections_count; edge_pos += WARP_SIZE)
        {
            if(edge_pos < connections_count)
            {
                const long long int global_edge_pos = edge_start + edge_pos;
                const int dst_id = _adjacent_ids[global_edge_pos];
                const int local_edge_pos = edge_pos;
                edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, frontier_pos);
            }
        }

        vertex_postprocess_op(src_id, frontier_pos, connections_count);
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
                                         VertexPostprocessOperation vertex_postprocess_op,
                                         int *_new_frontier_ids,
                                         bool _generate_frontier,
                                         int *_new_frontier_size)
{
    const int frontier_pos = blockIdx.x * blockDim.x + threadIdx.x + _vertex_part_start;

    if(frontier_pos < _vertex_part_end)
    {
        const int src_id = _frontier_ids[frontier_pos];

        const long long edge_start = _vertex_pointers[src_id];
        const int connections_count = _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];

        vertex_preprocess_op(src_id, frontier_pos, connections_count);

        for(register int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            if(edge_pos < connections_count)
            {
                const long long int global_edge_pos = edge_start + edge_pos;
                const int dst_id = _adjacent_ids[global_edge_pos];
                const int local_edge_pos = edge_pos;
                edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, frontier_pos);
            }
        }

        vertex_postprocess_op(src_id, frontier_pos, connections_count);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <int VirtualWarpSize, typename EdgeOperation, typename VertexPreprocessOperation, typename VertexPostprocessOperation>
void __global__ virtual_warp_per_vertex_kernel(const long long *_vertex_pointers,
                                               const int *_adjacent_ids,
                                               const int *_frontier_ids,
                                               const int _vertices_count,
                                               const int _vertex_part_start,
                                               const int _vertex_part_end,
                                               EdgeOperation edge_op,
                                               VertexPreprocessOperation vertex_preprocess_op,
                                               VertexPostprocessOperation vertex_postprocess_op,
                                               int *_new_frontier_ids,
                                               bool _generate_frontier,
                                               int *_new_frontier_size)
{
    const int virtual_warp_id = threadIdx.x / VirtualWarpSize;
    const int position_in_virtual_warp = threadIdx.x % VirtualWarpSize;

    const int frontier_pos = blockIdx.x * (blockDim.x / VirtualWarpSize) + virtual_warp_id + _vertex_part_start;

    if(frontier_pos < _vertex_part_end)
    {
        const int src_id = _frontier_ids[frontier_pos];

        const long long edge_start = _vertex_pointers[src_id];
        const int connections_count = _vertex_pointers[src_id + 1] - _vertex_pointers[src_id];

        vertex_preprocess_op(src_id, frontier_pos, connections_count);

        for(register int edge_pos = position_in_virtual_warp; edge_pos < connections_count; edge_pos += VirtualWarpSize)
        {
            if(edge_pos < connections_count)
            {
                const long long int global_edge_pos = edge_start + edge_pos;
                const int dst_id = _adjacent_ids[global_edge_pos];
                const int local_edge_pos = edge_pos;
                edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, frontier_pos);
            }
        }

        vertex_postprocess_op(src_id, frontier_pos, connections_count);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphPrimitivesGPU::advance_sparse(UndirectedGraph &_graph,
                                        FrontierGPU &_frontier,
                                        EdgeOperation edge_op,
                                        VertexPreprocessOperation vertex_preprocess_op,
                                        VertexPostprocessOperation vertex_postprocess_op,
                                        bool _generate_frontier)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    cudaDeviceSynchronize();
    double t1 = omp_get_wtime();
    #endif
    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);

    int grid_threshold_start = 0;
    int grid_threshold_end = 0;
    int block_threshold_start = 0;
    int block_threshold_end = 0;
    int warp_threshold_start = 0;
    int warp_threshold_end = 0;

    int vwp_16_threshold_start = 0;
    int vwp_16_threshold_end = 0;
    int vwp_8_threshold_start = 0;
    int vwp_8_threshold_end = 0;
    int vwp_4_threshold_start = 0;
    int vwp_4_threshold_end = 0;
    int vwp_2_threshold_start = 0;
    int vwp_2_threshold_end = 0;

    int thread_threshold_start = 0;
    int thread_threshold_end = 0;

    _frontier.split_sorted_frontier(vertex_pointers, grid_threshold_start, grid_threshold_end, block_threshold_start,
                                    block_threshold_end, warp_threshold_start, warp_threshold_end,
                                    vwp_16_threshold_start, vwp_16_threshold_end,
                                    vwp_8_threshold_start, vwp_8_threshold_end,
                                    vwp_4_threshold_start, vwp_4_threshold_end,
                                    vwp_2_threshold_start, vwp_2_threshold_end,
                                    thread_threshold_start, thread_threshold_end);


    int *tmp_new_frontier_buffer = _frontier.flags;
    int *new_frontier_size;
    if(_generate_frontier)
        MemoryAPI::allocate_managed_array(&new_frontier_size, 1);

    int grid_vertices_count = grid_threshold_end - grid_threshold_start;
    if (grid_vertices_count > 0)
    {
        grid_per_vertex_kernel <<< grid_vertices_count, 1, 0, grid_processing_stream >>>
                (vertex_pointers, adjacent_ids, _frontier.ids, vertices_count, grid_threshold_start,
                 grid_threshold_end, edge_op, vertex_preprocess_op, vertex_postprocess_op, tmp_new_frontier_buffer, _generate_frontier, new_frontier_size);
    }

    int block_vertices_count = block_threshold_end - block_threshold_start;
    if (block_vertices_count > 0)
    {
        block_per_vertex_kernel <<< block_vertices_count, BLOCK_SIZE, 0, block_processing_stream >>>
                (vertex_pointers, adjacent_ids, _frontier.ids, vertices_count, block_threshold_start,
                 block_threshold_end, edge_op, vertex_preprocess_op, vertex_postprocess_op, tmp_new_frontier_buffer, _generate_frontier, new_frontier_size);
    }

    int warp_vertices_count = warp_threshold_end - warp_threshold_start;
    if (warp_vertices_count > 0)
    {
        warp_per_vertex_kernel <<< WARP_SIZE*(warp_vertices_count - 1)/BLOCK_SIZE + 1, BLOCK_SIZE, 0, warp_processing_stream >>>
                (vertex_pointers, adjacent_ids, _frontier.ids, vertices_count, warp_threshold_start,
                 warp_threshold_end, edge_op, vertex_preprocess_op, vertex_postprocess_op, tmp_new_frontier_buffer, _generate_frontier, new_frontier_size);
    }

    int vwp_16_vertices_count = vwp_16_threshold_end - vwp_16_threshold_start;
    if(vwp_16_vertices_count > 0)
    {
        virtual_warp_per_vertex_kernel<16> <<< 16*(vwp_16_vertices_count - 1) / BLOCK_SIZE + 1, BLOCK_SIZE, 0, vwp_16_processing_stream >>>
                (vertex_pointers, adjacent_ids, _frontier.ids, vertices_count, vwp_16_threshold_start,
                 vwp_16_threshold_end, edge_op, vertex_preprocess_op, vertex_postprocess_op, tmp_new_frontier_buffer, _generate_frontier, new_frontier_size);
    }

    int vwp_8_vertices_count = vwp_8_threshold_end - vwp_8_threshold_start;
    if(vwp_8_vertices_count > 0)
    {
        virtual_warp_per_vertex_kernel<8> <<< 8*(vwp_8_vertices_count - 1) / BLOCK_SIZE + 1, BLOCK_SIZE, 0, vwp_8_processing_stream >>>
                (vertex_pointers, adjacent_ids, _frontier.ids, vertices_count, vwp_8_threshold_start,
                 vwp_8_threshold_end, edge_op, vertex_preprocess_op, vertex_postprocess_op, tmp_new_frontier_buffer, _generate_frontier, new_frontier_size);
    }

    int vwp_4_vertices_count = vwp_4_threshold_end - vwp_4_threshold_start;
    if(vwp_4_vertices_count > 0)
    {
        virtual_warp_per_vertex_kernel<4> <<< 4*(vwp_4_vertices_count - 1) / BLOCK_SIZE + 1, BLOCK_SIZE, 0, vwp_4_processing_stream >>>
                (vertex_pointers, adjacent_ids, _frontier.ids, vertices_count, vwp_4_threshold_start,
                 vwp_4_threshold_end, edge_op, vertex_preprocess_op, vertex_postprocess_op, tmp_new_frontier_buffer, _generate_frontier, new_frontier_size);
    }

    int vwp_2_vertices_count = vwp_2_threshold_end - vwp_2_threshold_start;
    if(vwp_2_vertices_count > 0)
    {
        virtual_warp_per_vertex_kernel<2> <<< 2*(vwp_2_vertices_count - 1) / BLOCK_SIZE + 1, BLOCK_SIZE, 0, vwp_2_processing_stream >>>
                (vertex_pointers, adjacent_ids, _frontier.ids, vertices_count, vwp_2_threshold_start,
                 vwp_2_threshold_end, edge_op, vertex_preprocess_op, vertex_postprocess_op, tmp_new_frontier_buffer, _generate_frontier, new_frontier_size);
    }

    int thread_vertices_count = thread_threshold_end - thread_threshold_start;
    if (thread_vertices_count > 0)
    {
        thread_per_vertex_kernel <<< (thread_vertices_count - 1) / BLOCK_SIZE + 1, BLOCK_SIZE, 0, thread_processing_stream >>>
                                                                                                  (vertex_pointers, adjacent_ids, _frontier.ids, vertices_count, thread_threshold_start,
                                                                                                          thread_threshold_end, edge_op, vertex_preprocess_op, vertex_postprocess_op, tmp_new_frontier_buffer, _generate_frontier, new_frontier_size);
    }
    cudaDeviceSynchronize();

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    double t2 = omp_get_wtime();
    cudaDeviceSynchronize();
    INNER_WALL_TIME += t2 - t1;
    INNER_ADVANCE_TIME += t2 - t1;
    int work = this->estimate_advance_work(_graph, _frontier);
    INNER_WALL_WORK += work;
    cout << "frontier size: " << _frontier.size() << "/" << vertices_count << ", " << 100.0*_frontier.size()/vertices_count << "%" << endl;
    cout << "advance time: " << (t2 - t1)*1000.0 << " ms" << endl;
    cout << "advance sparse BW: " << sizeof(int)*INT_ELEMENTS_PER_EDGE*work/((t2-t1)*1e9) << " GB/s" << endl << endl;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////