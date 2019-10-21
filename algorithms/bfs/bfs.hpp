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
void BFS<_TVertexValue, _TEdgeWeight>::gpu_direction_optimising_BFS(
                                         VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                         int *_levels,
                                         int _source_vertex)
{
    LOAD_VECTORISED_CSR_GRAPH_DATA(_graph)
    
    cout << "gpu bfs test" << endl;
    
    int *device_levels;
    SAFE_CALL(cudaMalloc((void**)&device_levels, vertices_count * sizeof(int)));
    
    gpu_direction_optimising_bfs_wrapper(first_part_ptrs, first_part_sizes,
                                         vector_segments_count,
                                         vector_group_ptrs, vector_group_sizes,
                                         outgoing_ids, number_of_vertices_in_first_part,
                                         device_levels, vertices_count, edges_count, _source_vertex);
    
    SAFE_CALL(cudaMemcpy(_levels, device_levels, vertices_count * sizeof(int), cudaMemcpyDeviceToHost));
    
    SAFE_CALL(cudaFree(device_levels));
}

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* bfs_hpp */
