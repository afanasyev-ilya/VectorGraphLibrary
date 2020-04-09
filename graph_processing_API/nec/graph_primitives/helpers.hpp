#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// performance debug variables
double INNER_WALL_NEC_TIME = 0;
double INNER_ADVANCE_NEC_TIME = 0;
double DETAILED_ADVANCE_PART_1_NEC_TIME = 0;
double DETAILED_ADVANCE_PART_2_NEC_TIME = 0;
double DETAILED_ADVANCE_PART_3_NEC_TIME = 0;
double INNER_COMPUTE_NEC_TIME = 0;
double INNER_GNF_NEC_TIME = 0;
double INNER_FILTER_NEC_TIME = 0;
double INNER_REDUCE_NEC_TIME = 0;
double INNER_PACK_NEC_TIME = 0;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline int get_vector_border(int _connections_count)
{
    return VECTOR_LENGTH * (int((_connections_count)/VECTOR_LENGTH));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void reset_nec_debug_timers()
{
    INNER_WALL_NEC_TIME = 0;
    INNER_ADVANCE_NEC_TIME = 0;
    INNER_COMPUTE_NEC_TIME = 0;
    INNER_GNF_NEC_TIME = 0;
    INNER_FILTER_NEC_TIME = 0;
    INNER_REDUCE_NEC_TIME = 0;
    INNER_PACK_NEC_TIME = 0;

    DETAILED_ADVANCE_PART_1_NEC_TIME = 0;
    DETAILED_ADVANCE_PART_2_NEC_TIME = 0;
    DETAILED_ADVANCE_PART_3_NEC_TIME = 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void print_nec_debug_timers(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph)
{
    cout << "INNER_WALL_NEC_TIME: " << INNER_WALL_NEC_TIME * 1000 << " ms" << endl;
    cout << "INNER perf: " << _graph.get_edges_count() / (INNER_WALL_NEC_TIME * 1e6) << " MTEPS" << endl;
    cout << "INNER_ADVANCE_NEC_TIME: " << int(100.0 * INNER_ADVANCE_NEC_TIME / INNER_WALL_NEC_TIME) << " %" << endl;
    cout << "INNER_COMPUTE_NEC_TIME: " << int(100.0 * INNER_COMPUTE_NEC_TIME / INNER_WALL_NEC_TIME) << " %" << endl;
    cout << "INNER_GNF_NEC_TIME: " << int(100.0 * INNER_GNF_NEC_TIME / INNER_WALL_NEC_TIME) << " %" << endl;
    cout << "INNER_REDUCE_NEC_TIME: " << int(100.0 * INNER_REDUCE_NEC_TIME / INNER_WALL_NEC_TIME) << " %" << endl;
    cout << "INNER_PACK_NEC_TIME: " << int(100.0 * INNER_PACK_NEC_TIME / INNER_WALL_NEC_TIME) << " %" << endl;
    cout << "INNER_FILTER_NEC_TIME: " << int(100.0 * INNER_FILTER_NEC_TIME / INNER_WALL_NEC_TIME) << " %" << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

