//
//  widest_paths.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 08/09/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef widest_paths_hpp
#define widest_paths_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void WidestPaths<_TVertexValue, _TEdgeWeight>::allocate_result_memory(int _vertices_count, _TEdgeWeight **_widths)
{
    *_widths = new _TEdgeWeight[_vertices_count];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void WidestPaths<_TVertexValue, _TEdgeWeight>::free_result_memory(_TEdgeWeight *_widths)
{
    delete[] _widths;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* widest_paths_hpp */
