/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FilterCondition, typename AbstractionClass>
void GraphAbstractions::common_generate_new_frontier(VGL_Graph &_graph,
                                                     VGL_Frontier &_frontier,
                                                     FilterCondition &&filter_cond,
                                                     AbstractionClass *_abstraction_class)
{
    _frontier.set_direction(current_traversal_direction);

    if((_graph.get_container_type() == VECTOR_CSR_GRAPH) && (_frontier.get_class_type() == VECTOR_CSR_FRONTIER))
    {
        VectorCSRGraph *container_graph = (VectorCSRGraph *)_graph.get_direction_data(current_traversal_direction);
        FrontierVectorCSR *container_frontier = (FrontierVectorCSR *)_frontier.get_container_data();

        _abstraction_class->generate_new_frontier_worker(*container_graph, *container_frontier, filter_cond);
    }
    else if(_graph.get_container_type() == EDGES_LIST_GRAPH)
    {
        EdgesListGraph *container_graph = (EdgesListGraph *)_graph.get_direction_data(current_traversal_direction);
        FrontierGeneral *container_frontier = (FrontierGeneral *)_frontier.get_container_data();

        _abstraction_class->generate_new_frontier_worker(*container_graph, *container_frontier, filter_cond);
    }
    else if(_graph.get_container_type() == CSR_GRAPH)
    {
        CSRGraph *container_graph = (CSRGraph *)_graph.get_direction_data(current_traversal_direction);
        FrontierGeneral *container_frontier = (FrontierGeneral *)_frontier.get_container_data();

        _abstraction_class->generate_new_frontier_worker(*container_graph, *container_frontier, filter_cond);
    }
    else
    {
        throw "Error: unsupported graph and frontier type in GraphAbstractions::generate_new_frontier";
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
