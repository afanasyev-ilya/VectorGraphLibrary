#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void gpu_dijkstra_all_active_wrapper(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, _TEdgeWeight *_distances,
                                     int _source_vertex, int &_iterations_count, AlgorithmTraversalType _traversal_direction);
template <typename _TVertexValue, typename _TEdgeWeight>
void gpu_dijkstra_partial_active_wrapper(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, _TEdgeWeight *_distances,
                                         int _source_vertex, int &_iterations_count);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
