//
//  change_state.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 30/06/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef change_state_h
#define change_state_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum StateOfBFS
{
    TOP_DOWN,
    BOTTOM_UP
};

enum GraphStructure
{
    POWER_LAW_GRAPH,
    UNIFORM_GRAPH
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define BOTTOM_UP_THRESHOLD 5
#define BOTTOM_UP_REMINDER_VERTEX -3

#define UNVISITED_VERTEX -1
#define ISOLATED_VERTEX -2

#define ENABLE_VECTOR_EXTENSION_THRESHOLD 0.1

#define POWER_LAW_EDGES_THRESHOLD 30

#define BOTTOM_UP_FORCE_SWITCH_THRESHOLD_POWER_LOW_GRAPHS 200

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
GraphStructure check_graph_structure(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph);

StateOfBFS change_state(int _current_queue_size, int _next_queue_size, int _vertices_count, long long _edges_count,
                        StateOfBFS _old_state, int _vis, int _in_lvl, bool &_use_vect_CSR_extension, int _cur_level,
                        GraphStructure _graph_structure);

StateOfBFS gpu_change_state(int _current_queue_size, int _next_queue_size, int _vertices_count, long long _edges_count,
                            StateOfBFS _old_state, int _vis, int _in_lvl, int _current_level, GraphStructure _graph_structure,
                            int _total_visited);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* change_state_h */
