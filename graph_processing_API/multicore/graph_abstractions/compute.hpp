#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ComputeOperation>
void GraphAbstractionsMulticore::compute_worker(UndirectedCSRGraph &_graph,
                                          FrontierMulticore &_frontier,
                                          ComputeOperation &&compute_op)
{
    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);
    if(_frontier.type == ALL_ACTIVE_FRONTIER)
    {
        int frontier_size = _frontier.max_size;

        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma omp for schedule(static)
        for(int src_id = 0; src_id < frontier_size; src_id++)
        {
            int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
            int vector_index = get_vector_index(src_id);
            compute_op(src_id, connections_count, vector_index);
        }
    }
    else if(_frontier.type == DENSE_FRONTIER)
    {
        int *frontier_flags = _frontier.flags;
        int frontier_size = _frontier.max_size;

        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma omp for schedule(static)
        for(int src_id = 0; src_id < frontier_size; src_id++)
        {
            if(frontier_flags[src_id] == IN_FRONTIER_FLAG)
            {
                int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
                int vector_index = get_vector_index(src_id);
                compute_op(src_id, connections_count, vector_index);
            }
        }
    }
    else if (_frontier.type == SPARSE_FRONTIER)
    {
        int frontier_size = _frontier.current_size;
        int *frontier_ids = _frontier.ids;

        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma omp for schedule(static)
        for(int frontier_pos = 0; frontier_pos < frontier_size; frontier_pos++)
        {
            int src_id = frontier_ids[frontier_pos];
            int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
            int vector_index = get_vector_index(src_id);
            compute_op(src_id, connections_count, vector_index);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ComputeOperation>
void GraphAbstractionsMulticore::compute(VectCSRGraph &_graph,
                                   FrontierMulticore &_frontier,
                                   ComputeOperation &&compute_op)
{
    Timer tm;
    tm.start();

    if(_frontier.get_direction() != current_traversal_direction)
    {
        throw "Error in GraphAbstractionsMulticore::compute : wrong frontier direction";
    }

    UndirectedCSRGraph *current_direction_graph;
    if(current_traversal_direction == SCATTER)
    {
        current_direction_graph = _graph.get_outgoing_graph_ptr();
    }
    else if(current_traversal_direction == GATHER)
    {
        current_direction_graph = _graph.get_incoming_graph_ptr();
    }

    if(omp_in_parallel())
    {
        #pragma omp barrier
        compute_worker(*current_direction_graph, _frontier, compute_op);
        #pragma omp barrier
    }
    else
    {
        #pragma omp parallel
        {
            compute_worker(*current_direction_graph, _frontier, compute_op);
        }
    }

    tm.end();
    performance_stats.update_compute_time(tm);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("Compute", _frontier.size(), COMPUTE_INT_ELEMENTS*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ComputeOperation>
void GraphAbstractionsMulticore::compute(ShardedCSRGraph &_graph,
                                   FrontierMulticore &_frontier,
                                   ComputeOperation &&compute_op)
{
    Timer tm;
    tm.start();

    if(_frontier.get_direction() != current_traversal_direction)
    {
        throw "Error in GraphAbstractionsMulticore::compute : wrong frontier direction";
    }

    UndirectedCSRGraph *current_direction_graph;
    if(current_traversal_direction == SCATTER)
    {
        current_direction_graph = _graph.get_outgoing_shard_ptr(0); // TODO
    }
    else if(current_traversal_direction == GATHER)
    {
        current_direction_graph = _graph.get_incoming_shard_ptr(0);  // TODO
    }

    if(omp_in_parallel())
    {
        #pragma omp barrier
        compute_worker(*current_direction_graph, _frontier, compute_op);
        #pragma omp barrier
    }
    else
    {
        #pragma omp parallel
        {
            compute_worker(*current_direction_graph, _frontier, compute_op);
        }
    }

    tm.end();
    performance_stats.update_compute_time(tm);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("Compute", _frontier.size(), COMPUTE_INT_ELEMENTS*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
