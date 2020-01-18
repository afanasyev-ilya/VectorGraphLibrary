//
//  bfs.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 03/06/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef bfs_hpp
#define bfs_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
BFS<_TVertexValue, _TEdgeWeight>::BFS(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph)
{
    int vertices_count = _graph.get_vertices_count();
    
    #ifdef __USE_NEC_SX_AURORA_TSUBASA__
    active_vertices_buffer = _graph.template vertex_array_alloc<int>();
    active_ids = _graph.template vertex_array_alloc<int>();
    
    #pragma omp parallel
    {}
    #endif
    
    #ifdef __USE_GPU__
    cudaMalloc((void**)&active_ids, sizeof(int) * vertices_count);
    cudaMalloc((void**)&active_vertices_buffer, sizeof(int) * vertices_count);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
BFS<_TVertexValue, _TEdgeWeight>::~BFS()
{
    #ifdef __USE_NEC_SX_AURORA_TSUBASA__
    if(active_vertices_buffer != NULL)
    {
        delete []active_vertices_buffer;
    }
    if(active_ids != NULL)
    {
        delete []active_ids;
    }
    #endif
    
    #ifdef __USE_GPU__
    cudaFree(active_ids);
    cudaFree(active_vertices_buffer);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::allocate_result_memory(int _vertices_count, int **_levels)
{
    *_levels = new int[_vertices_count];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::free_result_memory(int *_levels)
{
    delete[] _levels;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::allocate_device_result_memory(int _vertices_count, int **_device_levels)
{
    cudaMalloc((void**)_device_levels, _vertices_count * sizeof(int));
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::free_device_result_memory(int *_device_levels)
{
    cudaFree(_device_levels);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::copy_result_to_host(int *_host_levels, int *_device_levels, int _vertices_count)
{
    cudaMemcpy(_host_levels, _device_levels, _vertices_count * sizeof(int), cudaMemcpyDeviceToHost);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::gpu_direction_optimising_BFS(
                                         ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                         int *_device_levels,
                                         int _source_vertex)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    
    _graph.move_to_device();
    
    int iterations_count = 0;
    gpu_direction_optimising_bfs_wrapper<_TVertexValue, _TEdgeWeight>(_graph, _device_levels, _source_vertex, iterations_count,
                                                                      active_ids, active_vertices_buffer);
    
    //cudaMemcpy(_levels, device_levels, vertices_count * sizeof(int), cudaMemcpyDeviceToHost);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO perf stats

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* bfs_hpp */
