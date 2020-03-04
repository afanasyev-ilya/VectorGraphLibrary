//
//  widest_paths.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 08/09/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef widest_paths_h
#define widest_paths_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class WidestPaths
{
public:
    static void allocate_result_memory(int _vertices_count, _TEdgeWeight **_widths);
    static void free_result_memory    (_TEdgeWeight *_widths);

    static void bellman_ford(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_reversed_graph,
                             int _source_vertex, _TEdgeWeight *_widths);
    
    static void bellman_ford(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_reversed_graph,
                             int _source_vertex, _TEdgeWeight *_widths);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "widest_paths.hpp"
#include "bellman_ford.hpp"

#endif /* widest_paths_h */
