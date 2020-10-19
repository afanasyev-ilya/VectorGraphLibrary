#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GraphAbstractionsGPU::GraphAbstractionsGPU(VectCSRGraph &_graph, TraversalDirection _initial_traversal)
{
    processed_graph_ptr = &_graph;
    current_traversal_direction = _initial_traversal;
    direction_shift = _graph.get_edges_count();

    cudaStreamCreate(&grid_processing_stream);
    cudaStreamCreate(&block_processing_stream);
    cudaStreamCreate(&warp_processing_stream);
    cudaStreamCreate(&vwp_16_processing_stream);
    cudaStreamCreate(&vwp_8_processing_stream);
    cudaStreamCreate(&vwp_4_processing_stream);
    cudaStreamCreate(&vwp_2_processing_stream);
    cudaStreamCreate(&thread_processing_stream);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GraphAbstractionsGPU::GraphAbstractionsGPU(ShardedCSRGraph &_graph, TraversalDirection _initial_traversal)
{
    processed_graph_ptr = NULL; // TODO
    current_traversal_direction = _initial_traversal;
    direction_shift = _graph.get_edges_count(); // TODO

    cudaStreamCreate(&grid_processing_stream);
    cudaStreamCreate(&block_processing_stream);
    cudaStreamCreate(&warp_processing_stream);
    cudaStreamCreate(&vwp_16_processing_stream);
    cudaStreamCreate(&vwp_8_processing_stream);
    cudaStreamCreate(&vwp_4_processing_stream);
    cudaStreamCreate(&vwp_2_processing_stream);
    cudaStreamCreate(&thread_processing_stream);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GraphAbstractionsGPU::~GraphAbstractionsGPU()
{
    cudaStreamDestroy(block_processing_stream);
    cudaStreamDestroy(warp_processing_stream);
    cudaStreamDestroy(thread_processing_stream);
    cudaStreamDestroy(grid_processing_stream);
    cudaStreamDestroy(vwp_16_processing_stream);
    cudaStreamDestroy(vwp_8_processing_stream);
    cudaStreamDestroy(vwp_4_processing_stream);
    cudaStreamDestroy(vwp_2_processing_stream);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
