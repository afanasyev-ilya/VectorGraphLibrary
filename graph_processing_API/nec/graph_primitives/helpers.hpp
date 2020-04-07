#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

auto EMPTY_EDGE_OP = [] (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int vector_index,
                         DelayedWriteNEC &delayed_write) {};
auto EMPTY_VERTEX_OP = [] (int src_id, int connections_count, int vector_index, DelayedWriteNEC &delayed_write){};

auto ALL_ACTIVE_FRONTIER_CONDITION = [] (int src_id)->int
{
    return NEC_IN_FRONTIER_FLAG;
};

auto EMPTY_COMPUTE_OP = [] (int src_id, int connections_count, int vector_index) {};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// performance debug variables
double INNER_WALL_NEC_TIME = 0;
double INNER_ADVANCE_NEC_TIME = 0;
double INNER_COMPUTE_NEC_TIME = 0;
double INNER_GNF_NEC_TIME = 0;
double INNER_FILTER_NEC_TIME = 0;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline int get_vector_connection_border(int _connections_count)
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
    cout << "INNER_FILTER_NEC_TIME: " << int(100.0 * INNER_FILTER_NEC_TIME / INNER_WALL_NEC_TIME) << " %" << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

