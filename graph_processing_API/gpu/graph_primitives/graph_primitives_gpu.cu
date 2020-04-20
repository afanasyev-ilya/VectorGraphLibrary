#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GraphPrimitivesGPU::GraphPrimitivesGPU()
{
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

GraphPrimitivesGPU::~GraphPrimitivesGPU()
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
