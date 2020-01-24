#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "traversal_kernels.cu"
#include "init_kernels.cu"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GraphPrimitivesGPU::GraphPrimitivesGPU()
{
    cudaStreamCreate(&grid_processing_stream);
    cudaStreamCreate(&block_processing_stream);
    cudaStreamCreate(&warp_processing_stream);
    cudaStreamCreate(&thread_processing_stream);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GraphPrimitivesGPU::~GraphPrimitivesGPU()
{
    cudaStreamDestroy(block_processing_stream);
    cudaStreamDestroy(warp_processing_stream);
    cudaStreamDestroy(thread_processing_stream);
    cudaStreamDestroy(grid_processing_stream);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename InitOperation>
void GraphPrimitivesGPU::init(int _size, InitOperation init_op)
{
    init_kernel <<< (_size - 1)/BLOCK_SIZE + 1, BLOCK_SIZE >>> (_size, init_op);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphPrimitivesGPU::advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                 FrontierGPU &_frontier,
                                 EdgeOperation edge_op,
                                 VertexPreprocessOperation vertex_preprocess_op,
                                 VertexPostprocessOperation vertex_postprocess_op) {
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
                (outgoing_ptrs, outgoing_ids, _frontier.frontier_ids, vertices_count, grid_threshold_start,
                 grid_threshold_end, edge_op, vertex_preprocess_op, vertex_postprocess_op);
    }

    int block_vertices_count = block_threshold_end - block_threshold_start;
    if (block_vertices_count > 0)
    {
        block_per_vertex_kernel <<< block_vertices_count, BLOCK_SIZE, 0, block_processing_stream >>>
               (outgoing_ptrs, outgoing_ids, _frontier.frontier_ids, vertices_count, block_threshold_start,
                block_threshold_end, edge_op, vertex_preprocess_op, vertex_postprocess_op);
    }

    int warp_vertices_count = warp_threshold_end - warp_threshold_start;
    if (warp_vertices_count > 0)
    {
        warp_per_vertex_kernel <<< warp_vertices_count, WARP_SIZE, 0, warp_processing_stream >>>
              (outgoing_ptrs, outgoing_ids, _frontier.frontier_ids, vertices_count, warp_threshold_start,
               warp_threshold_end, edge_op, vertex_preprocess_op, vertex_postprocess_op);
    }

    int thread_vertices_count = thread_threshold_end - thread_threshold_start;
    if (thread_vertices_count)
    {
        thread_per_vertex_kernel <<< (thread_vertices_count - 1) / BLOCK_SIZE + 1, BLOCK_SIZE, 0, thread_processing_stream >>>
                (outgoing_ptrs, outgoing_ids, _frontier.frontier_ids, vertices_count, thread_threshold_start,
                 thread_threshold_end, edge_op, vertex_preprocess_op, vertex_postprocess_op);
    }
    cudaDeviceSynchronize();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
