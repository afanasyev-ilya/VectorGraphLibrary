//
//  label_propagation.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 14/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef label_propagation_h
#define label_propagation_h

#include <map>

#ifdef __USE_NEC_SX_AURORA__
//#include <ftrace.h>
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class LabelPropagation
{
private:
    static void process_reminder_vertices(int *_outgoing_ids, int *_labels, long long int *_vector_group_ptrs,
                                          int *_vector_group_sizes, int _vector_segments_count, int _reminder_start,
                                          int _threads_count, int _max_dict_size, VectorDictionary **_vector_dicts);
public:
    static void allocate_result_memory  (int _vertices_count, int **_result);
    static void free_result_memory      (int *_result);

    static void label_propagation_map_based(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_labels,
                                            int _threads_count = 8);
    
    static void label_propagation_vector_map_based(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_labels,
                                                   int _threads_count = 8);
    
    static void append_graph_with_labels(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_labels);
    
    static void analyse_result(int *_labels, int _vertices_count);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "label_propagation.hpp"
#include "vect_dict_based_lp.hpp"
#include "map_based_lp.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* label_propagation_h */
