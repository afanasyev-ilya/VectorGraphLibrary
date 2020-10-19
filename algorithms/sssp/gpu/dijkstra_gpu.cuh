#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void gpu_dijkstra_all_active_wrapper(VectCSRGraph &_graph, EdgesArray<_T> &_weights, VerticesArray<_T> &_distances,
                                     int _source_vertex, int &_iterations_count, AlgorithmTraversalType _traversal_direction);

template <typename _T>
void gpu_dijkstra_partial_active_wrapper(VectCSRGraph &_graph, EdgesArray<_T> &_weights, VerticesArray<_T> &_distances,
                                         int _source_vertex, int &_iterations_count);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
