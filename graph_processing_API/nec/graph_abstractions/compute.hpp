#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ComputeOperation, typename Graph_Container>
void GraphAbstractionsNEC::compute_worker(Graph_Container &_graph,
                                          VGL_Frontier &_frontier,
                                          ComputeOperation &&compute_op)
{
    int frontier_size = _frontier.get_size();
    int *frontier_flags = _frontier.get_flags();
    int *frontier_ids = _frontier.get_ids();
    FrontierSparsityType frontier_type = _frontier.get_sparsity_type();

    if(frontier_type == ALL_ACTIVE_FRONTIER)
    {
        #pragma _NEC cncall
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma omp for schedule(static)
        for(int src_id = 0; src_id < frontier_size; src_id++)
        {
            int connections_count = _graph.get_connections_count(src_id);
            int vector_index = get_vector_index(src_id);
            compute_op(src_id, connections_count, vector_index);
        }
    }
    else if(frontier_type == DENSE_FRONTIER) // TODO check
    {
        #pragma _NEC cncall
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
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
        #pragma _NEC cncall
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
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
void GraphAbstractionsNEC::compute_container_call(VGL_Graph &_graph,
                                                  VGL_Frontier &_frontier,
                                                  ComputeOperation &&compute_op)
{
    if(_graph.get_container_type() == VECTOR_CSR_GRAPH)
    {
        VectorCSRGraph *container_graph = (VectorCSRGraph *)_graph.get_direction_data(current_traversal_direction);
        compute_worker(*container_graph, _frontier, compute_op);
    }
    else if(_graph.get_container_type() == EDGES_LIST_GRAPH)
    {
        EdgesListGraph *container_graph = (EdgesListGraph *)_graph.get_direction_data(current_traversal_direction);
        compute_worker(*container_graph, _frontier, compute_op);
    }
    else
    {
        throw "Error in GraphAbstractionsNEC::compute : unsupported container type";
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ComputeOperation>
void GraphAbstractionsNEC::compute(VGL_Graph &_graph,
                                   VGL_Frontier &_frontier,
                                   ComputeOperation &&compute_op)
{
    Timer tm;
    tm.start();

    if(_frontier.get_direction() != current_traversal_direction) // TODO check
    {
        throw "Error in GraphAbstractionsNEC::compute : wrong frontier direction";
    }

    if(omp_in_parallel())
    {
        #pragma omp barrier
        compute_container_call(_graph, _frontier, compute_op);
        #pragma omp barrier
    }
    else
    {
        #pragma omp parallel
        {
            compute_container_call(_graph, _frontier, compute_op);
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
