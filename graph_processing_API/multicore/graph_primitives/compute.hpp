#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename ComputeOperation>
void GraphPrimitivesMulticore::compute(ExtendedCSRGraph &_graph,
                                       FrontierMulticore &_frontier,
                                       ComputeOperation compute_op)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    const long long int *vertex_pointers = vertex_pointers;

    int max_frontier_size = _frontier.max_size;

    #pragma ivdep
    #pragma simd
    #pragma omp parallel for schedule(static)
    for(int src_id = 0; src_id < max_frontier_size; src_id++)
    {
        int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
        int vector_index = get_vector_index(src_id);
        compute_op(src_id, connections_count, vector_index);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

