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
BFS<_TVertexValue, _TEdgeWeight>::BFS()
{
    active_vertices_buffer = NULL;
    vectorised_outgoing_ids = NULL;
    active_ids = NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
BFS<_TVertexValue, _TEdgeWeight>::~BFS()
{
    if(active_vertices_buffer != NULL)
    {
        delete []active_vertices_buffer;
    }
    if(vectorised_outgoing_ids != NULL)
    {
        delete []vectorised_outgoing_ids;
    }
    if(active_ids != NULL)
    {
        delete []active_ids;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::init_temporary_datastructures(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph)
{
    active_vertices_buffer = _graph.template vertex_array_alloc<int>();
    
    int vertices_count       = _graph.get_vertices_count();
    int *outgoing_ids        = _graph.get_outgoing_ids    ();
    long long *outgoing_ptrs = _graph.get_outgoing_ptrs   ();
    
    int zero_nodes_count = 0;
    #pragma _NEC vector
    #pragma omp parallel for schedule(static) reduction(+: zero_nodes_count)
    for(int src_id = 0; src_id < vertices_count; src_id++)
    {
        int connections = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
        if(connections == 0)
        {
            zero_nodes_count++;
        }
    }
    int non_zero_vertices_count = vertices_count - zero_nodes_count;
    
    vectorised_outgoing_ids = new int[non_zero_vertices_count * BOTTOM_UP_THRESHOLD];
    
    for(int step = 0; step < BOTTOM_UP_THRESHOLD; step++)
    {
        #pragma omp parallel for schedule(static)
        for(int src_id = 0; src_id < non_zero_vertices_count; src_id++)
        {
            int connections = outgoing_ptrs[src_id + 1] - outgoing_ptrs[src_id];
            long long start_pos = outgoing_ptrs[src_id];
                
            if(step < connections)
            {
                int shift = step;
                int dst_id = outgoing_ids[start_pos + shift];
                vectorised_outgoing_ids[src_id + non_zero_vertices_count * step] = dst_id;
            }
        }
    }
    
    active_ids = _graph.template vertex_array_alloc<int>();
    
    #pragma omp parallel
    {}
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
void BFS<_TVertexValue, _TEdgeWeight>::gpu_direction_optimising_BFS(
                                         VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                         int *_levels,
                                         int _source_vertex)
{
    LOAD_VECTORISED_CSR_GRAPH_DATA(_graph)
    
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
