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

    #pragma omp parallel for schedule(static)
    for(int src_id = 0; src_id < max_frontier_size; src_id++)
    {
        int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
        int vector_index = 0;//get_vector_index(src_id);
        compute_op(src_id, connections_count, vector_index);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

