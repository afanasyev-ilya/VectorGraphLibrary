/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ComputeOperation, typename AbstractionClass>
void GraphAbstractions::compute_container_call(VGL_Graph &_graph,
                                               VGL_Frontier &_frontier,
                                               ComputeOperation &&compute_op,
                                               AbstractionClass *_abstraction_class)
{
    if(_graph.get_container_type() == VECTOR_CSR_GRAPH)
    {
        VectorCSRGraph *container_graph = (VectorCSRGraph *)_graph.get_direction_data(current_traversal_direction);
        FrontierVectorCSR *container_frontier = (FrontierVectorCSR *)_frontier.get_container_data();

        #ifdef __USE_GPU__
        _abstraction_class->compute_worker(*container_graph, *container_frontier, compute_op);
        #else
        OMP_PARALLEL_CALL((_abstraction_class->compute_worker(*container_graph, *container_frontier, compute_op)));
        #endif
    }
    else if(_graph.get_container_type() == EDGES_LIST_GRAPH)
    {
        EdgesListGraph *container_graph = (EdgesListGraph *)_graph.get_direction_data(current_traversal_direction);
        FrontierGeneral *container_frontier = (FrontierGeneral *)_frontier.get_container_data();

        #ifdef __USE_GPU__
        _abstraction_class->compute_worker(*container_graph, *container_frontier, compute_op);
        #else
        OMP_PARALLEL_CALL((_abstraction_class->compute_worker(*container_graph, *container_frontier, compute_op)));
        #endif
    }
    else if(_graph.get_container_type() == CSR_GRAPH)
    {
        CSRGraph *container_graph = (CSRGraph *)_graph.get_direction_data(current_traversal_direction);
        FrontierGeneral *container_frontier = (FrontierGeneral *)_frontier.get_container_data();

        #ifdef __USE_GPU__
        _abstraction_class->compute_worker(*container_graph, *container_frontier, compute_op);
        #else
        OMP_PARALLEL_CALL((_abstraction_class->compute_worker(*container_graph, *container_frontier, compute_op)));
        #endif
    }
    else
    {
        throw "Error in GraphAbstractions::compute : unsupported container type";
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ComputeOperation, typename AbstractionClass>
void GraphAbstractions::common_compute(VGL_Graph &_graph,
                                       VGL_Frontier &_frontier,
                                       ComputeOperation &&compute_op,
                                       AbstractionClass *_abstraction_class)
{
    Timer tm;
    tm.start();

    if(_frontier.get_direction() != current_traversal_direction)
    {
        throw "Error in GraphAbstractions::compute : wrong frontier direction";
    }

    compute_container_call(_graph, _frontier, compute_op, _abstraction_class);

    tm.end();
    long long work = _frontier.size();
    performance_stats.update_compute_time(tm);
    performance_stats.update_bytes_requested(COMPUTE_INT_ELEMENTS*sizeof(int)*work);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("Compute", work, COMPUTE_INT_ELEMENTS*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
