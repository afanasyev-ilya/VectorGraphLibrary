#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ComputeOperation>
void GraphAbstractionsMulticore::compute(VectCSRGraph &_graph,
                                         FrontierMulticore &_frontier,
                                         ComputeOperation compute_op)
{
    UndirectedCSRGraph *current_direction_graph = _graph.get_direction_graph_ptr(SCATTER);
    LOAD_UNDIRECTED_CSR_GRAPH_DATA((*current_direction_graph));

    int max_frontier_size = _frontier.max_size;

    Timer tm;
    tm.start();
    #pragma simd
    #pragma ivdep
    #pragma omp parallel for
    for(int src_id = 0; src_id < max_frontier_size; src_id++)
    {
        int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
        int vector_index = src_id % VECTOR_LENGTH;
        compute_op(src_id, connections_count, vector_index);
    }
    tm.end();
    performance_stats.update_compute_time(tm);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_and_bandwidth_stats("Compute", _frontier.size(), COMPUTE_INT_ELEMENTS*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

