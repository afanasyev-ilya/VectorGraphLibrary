//
//  cc.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 05/09/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef cc_hpp
#define cc_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ConnectedComponents<_TVertexValue, _TEdgeWeight>::allocate_result_memory(int _vertices_count, int **_cc_result)
{
    *_cc_result = new int[_vertices_count];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ConnectedComponents<_TVertexValue, _TEdgeWeight>::free_result_memory(int *_cc_result)
{
    delete[] _cc_result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* cc_hpp */
