/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename ReduceOperation, typename AbstractionClass>
void GraphAbstractions::common_reduce(VGL_Graph &_graph,
                                      VGL_Frontier &_frontier,
                                      ReduceOperation &&reduce_op,
                                      REDUCE_TYPE _reduce_type,
                                      _T &_result,
                                      AbstractionClass *_abstraction_class)
{
    Timer tm;
    tm.start();

    if(_frontier.get_direction() != current_traversal_direction)
    {
        throw "Error in GraphAbstractions::reduce : wrong frontier direction";
    }

    if(omp_in_parallel())
    {
        throw "Error in GraphAbstractions::reduce : reduce can not be called in parallel region (reduction construct)";
    }

    _result = 0;

    if(_reduce_type == REDUCE_SUM)
    {
        if(_graph.get_container_type() == VECTOR_CSR_GRAPH)
        {
            VectorCSRGraph *container_graph = (VectorCSRGraph *)_graph.get_direction_data(current_traversal_direction);
            FrontierVectorCSR *container_frontier = (FrontierVectorCSR *)_frontier.get_container_data();

            _abstraction_class->reduce_worker_sum(*container_graph, *container_frontier, reduce_op, _result);
        }
        else if(_graph.get_container_type() == EDGES_LIST_GRAPH)
        {
            EdgesListGraph *container_graph = (EdgesListGraph *)_graph.get_direction_data(current_traversal_direction);
            FrontierEdgesList *container_frontier = (FrontierEdgesList *)_frontier.get_container_data();

            _abstraction_class->reduce_worker_sum(*container_graph, *container_frontier, reduce_op, _result);
        }
        else if(_graph.get_container_type() == CSR_GRAPH)
        {
            CSRGraph *container_graph = (CSRGraph *)_graph.get_direction_data(current_traversal_direction);
            FrontierCSR *container_frontier = (FrontierCSR *)_frontier.get_container_data();

            _abstraction_class->reduce_worker_sum(*container_graph, *container_frontier, reduce_op, _result);
        }
        else
        {
            throw "Error in GraphAbstractions::compute : unsupported container type";
        }
    }
    else
    {
        throw "Error in GraphAbstractions::reduce: non-sum reduce are currently unsupported";
    }

    tm.end();
    long long work = _frontier.size();
    performance_stats.update_reduce_time(tm);
    performance_stats.update_bytes_requested(REDUCE_INT_ELEMENTS*sizeof(int)*work);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("Reduce", work, REDUCE_INT_ELEMENTS*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
