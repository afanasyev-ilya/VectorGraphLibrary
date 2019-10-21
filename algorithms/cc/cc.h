//
//  cc.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 05/09/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef cc_h
#define cc_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_INTEL__
#include <limits>
#include <cfloat>
#endif

#ifdef __USE_NEC_SX_AURORA__
#include <ftrace.h>
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class ConnectedComponents
{
public:
    static void allocate_result_memory(int _vertices_count, int **_cc_result);
    static void free_result_memory    (int *_cc_result);
    
    static void nec_shiloach_vishkin(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_cc_result);
    static void nec_shiloach_vishkin(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_cc_result);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cc.hpp"
#include "shiloach_vishkin.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* cc_h */
