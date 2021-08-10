#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ComputeOperation, typename GraphContainer, typename FrontierContainer>
void GraphAbstractionsMulticore::compute_worker(GraphContainer &_graph,
                                                FrontierContainer &_frontier,
                                                ComputeOperation &&compute_op)
{
    int frontier_size = _frontier.get_size();
    int *frontier_flags = _frontier.get_flags();
    int *frontier_ids = _frontier.get_ids();
    FrontierSparsityType frontier_type = _frontier.get_sparsity_type();

    if(frontier_type == ALL_ACTIVE_FRONTIER)
    {
        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma omp for schedule(static)
        for(int src_id = 0; src_id < frontier_size; src_id++)
        {
            int connections_count = _graph.get_connections_count(src_id);
            int vector_index = get_vector_index(src_id);
            compute_op(src_id, connections_count, vector_index);
        }
    }
    else if(frontier_type == DENSE_FRONTIER)
    {
        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma omp for schedule(static)
        for(int src_id = 0; src_id < frontier_size; src_id++)
        {
            if(frontier_flags[src_id] == IN_FRONTIER_FLAG)
            {
                int connections_count = _graph.get_connections_count(src_id);
                int vector_index = get_vector_index(src_id);
                compute_op(src_id, connections_count, vector_index);
            }
        }
    }
    else if (frontier_type == SPARSE_FRONTIER)
    {
        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma omp for schedule(static)
        for(int frontier_pos = 0; frontier_pos < frontier_size; frontier_pos++)
        {
            int src_id = frontier_ids[frontier_pos];
            int connections_count = _graph.get_connections_count(src_id);
            int vector_index = get_vector_index(src_id);
            compute_op(src_id, connections_count, vector_index);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ComputeOperation>
void GraphAbstractionsMulticore::compute(VGL_Graph &_graph,
                                         VGL_Frontier &_frontier,
                                         ComputeOperation &&compute_op)
{
    this->common_compute(_graph, _frontier, compute_op, this);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
