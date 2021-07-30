#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ComputeOperation>
void GraphAbstractionsMulticore::compute_worker(VGL_Graph &_graph,
                                                VGL_Frontier &_frontier,
                                                ComputeOperation &&compute_op)
{
    UndirectedGraph *current_direction_graph = _graph.get_direction_data(current_traversal_direction);

    int frontier_size = _frontier.get_size();
    int *frontier_flags = _frontier.get_flags();
    int *frontier_ids = _frontier.get_ids();
    FrontierType frontier_type = _frontier.get_type();

    if(frontier_type == ALL_ACTIVE_FRONTIER)
    {
        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma omp for schedule(static)
        for(int src_id = 0; src_id < frontier_size; src_id++)
        {
            int connections_count = current_direction_graph->get_connections_count(src_id);
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
                int connections_count = current_direction_graph->get_connections_count(src_id);
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
            int connections_count = current_direction_graph->get_connections_count(src_id);
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
    Timer tm;
    tm.start();

    if(_frontier.get_direction() != current_traversal_direction)
    {
        throw "Error in GraphAbstractionsMulticore::compute : wrong frontier direction";
    }

    if(omp_in_parallel())
    {
        #pragma omp barrier
        compute_worker(_graph, _frontier, compute_op);
        #pragma omp barrier
    }
    else
    {
        #pragma omp parallel
        {
            compute_worker(_graph, _frontier, compute_op);
        }
    }

    tm.end();
    long long work = _frontier.size();
    performance_stats.update_compute_time(tm);
    performance_stats.update_bytes_requested(COMPUTE_INT_ELEMENTS*sizeof(int)*work);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("Compute", work, COMPUTE_INT_ELEMENTS*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
