#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void gpu_dijkstra_all_active_wrapper(UndirectedCSRGraph &_graph, _TEdgeWeight *_distances,
                                     int _source_vertex, int &_iterations_count, AlgorithmTraversalType _traversal_direction);

void gpu_dijkstra_partial_active_wrapper(UndirectedCSRGraph &_graph, _TEdgeWeight *_distances,
                                         int _source_vertex, int &_iterations_count);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
