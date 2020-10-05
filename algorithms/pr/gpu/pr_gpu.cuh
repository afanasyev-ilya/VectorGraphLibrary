#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void page_rank_wrapper(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                       float *_page_ranks,
                       float _convergence_factor,
                       int _max_iterations,
                       int &_iterations_count,
                       AlgorithmTraversalType _traversal_direction);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
