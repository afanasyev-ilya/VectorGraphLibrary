/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation, typename AbstractionClass>
void GraphAbstractions::common_scatter(VGL_Graph &_graph,
                                       VGL_Frontier &_frontier,
                                       EdgeOperation &&edge_op,
                                       VertexPreprocessOperation &&vertex_preprocess_op,
                                       VertexPostprocessOperation &&vertex_postprocess_op,
                                       CollectiveEdgeOperation &&collective_edge_op,
                                       CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                                       CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op,
                                       AbstractionClass *_abstraction_class)
{
    Timer tm;
    tm.start();

    if(current_traversal_direction != SCATTER)
    {
        throw "Error in GraphAbstractions::scatter : wrong traversal direction";
    }
    if(_frontier.get_direction() != current_traversal_direction)
    {
        throw "Error in GraphAbstractions::scatter : wrong frontier direction";
    }

    bool inner_mpi_processing = false;
    #ifdef __USE_MPI__
    inner_mpi_processing = true;
    #endif
    if(_graph.get_container_type() == VECTOR_CSR_GRAPH)
    {
        VectorCSRGraph *container_graph = (VectorCSRGraph *)_graph.get_outgoing_data();
        FrontierVectorCSR *container_frontier = (FrontierVectorCSR *)_frontier.get_container_data();

        #ifdef __USE_GPU__
        _abstraction_class->advance_worker(*container_graph, *container_frontier, edge_op,
                                           vertex_preprocess_op, vertex_postprocess_op,
                                           collective_edge_op, collective_vertex_preprocess_op,
                                           collective_vertex_postprocess_op,
                                           inner_mpi_processing);
        #else
        OMP_PARALLEL_CALL((_abstraction_class->advance_worker(*container_graph, *container_frontier, edge_op,
                                                              vertex_preprocess_op, vertex_postprocess_op,
                                                              collective_edge_op, collective_vertex_preprocess_op,
                                                              collective_vertex_postprocess_op,
                                                              inner_mpi_processing)));
        #endif
    }
    else if(_graph.get_container_type() == EDGES_LIST_GRAPH)
    {
        EdgesListGraph *container_graph = (EdgesListGraph *)_graph.get_outgoing_data();
        FrontierEdgesList *container_frontier = (FrontierEdgesList *)_frontier.get_container_data();

        #ifdef __USE_GPU__
        _abstraction_class->advance_worker(*container_graph, *container_frontier, edge_op,
                                           vertex_preprocess_op, vertex_postprocess_op,
                                           collective_edge_op, collective_vertex_preprocess_op,
                                           collective_vertex_postprocess_op,
                                           inner_mpi_processing);
        #else
        OMP_PARALLEL_CALL((_abstraction_class->advance_worker(*container_graph, *container_frontier, edge_op,
                                                              vertex_preprocess_op, vertex_postprocess_op,
                                                              collective_edge_op, collective_vertex_preprocess_op,
                                                              collective_vertex_postprocess_op,
                                                              inner_mpi_processing)));
        #endif
    }
    else if(_graph.get_container_type() == CSR_GRAPH)
    {
        CSRGraph *container_graph = (CSRGraph *)_graph.get_outgoing_data();
        FrontierCSR *container_frontier = (FrontierCSR *)_frontier.get_container_data();

        #ifdef __USE_GPU__
        _abstraction_class->advance_worker(*container_graph, *container_frontier, edge_op,
                                           vertex_preprocess_op, vertex_postprocess_op,
                                           collective_edge_op, collective_vertex_preprocess_op,
                                           collective_vertex_postprocess_op,
                                           inner_mpi_processing);
        #else
        OMP_PARALLEL_CALL((_abstraction_class->advance_worker(*container_graph, *container_frontier, edge_op,
                                                              vertex_preprocess_op, vertex_postprocess_op,
                                                              collective_edge_op, collective_vertex_preprocess_op,
                                                              collective_vertex_postprocess_op,
                                                              inner_mpi_processing)));
        #endif
    }
    else if(_graph.get_container_type() == CSR_VG_GRAPH)
    {
        CSR_VG_Graph *container_graph = (CSR_VG_Graph *)_graph.get_outgoing_data();
        FrontierCSR_VG *container_frontier = (FrontierCSR_VG *)_frontier.get_container_data();

        #ifdef __USE_GPU__
        _abstraction_class->advance_worker(*container_graph, *container_frontier, edge_op,
                                           vertex_preprocess_op, vertex_postprocess_op,
                                           collective_edge_op, collective_vertex_preprocess_op,
                                           collective_vertex_postprocess_op,
                                           inner_mpi_processing);
        #else
        OMP_PARALLEL_CALL((_abstraction_class->advance_worker(*container_graph, *container_frontier, edge_op,
                                                              vertex_preprocess_op, vertex_postprocess_op,
                                                              collective_edge_op, collective_vertex_preprocess_op,
                                                              collective_vertex_postprocess_op,
                                                              inner_mpi_processing)));
        #endif
    }
    else
    {
        throw "Error in GraphAbstractions::scatter unsupported graph type";
    }

    tm.end();
    performance_stats.update_scatter_time(tm);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation, typename AbstractionClass>
void GraphAbstractions::common_gather(VGL_Graph &_graph,
                                      VGL_Frontier &_frontier,
                                      EdgeOperation &&edge_op,
                                      VertexPreprocessOperation &&vertex_preprocess_op,
                                      VertexPostprocessOperation &&vertex_postprocess_op,
                                      CollectiveEdgeOperation &&collective_edge_op,
                                      CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                                      CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op,
                                      AbstractionClass *_abstraction_class)
{
    Timer tm;
    tm.start();

    if(current_traversal_direction != GATHER)
    {
        throw "Error in GraphAbstractions::gather : wrong traversal direction";
    }
    if(_frontier.get_direction() != current_traversal_direction)
    {
        throw "Error in GraphAbstractions::gather : wrong frontier direction";
    }

    bool inner_mpi_processing = false;
    #ifdef __USE_MPI__
    inner_mpi_processing = true;
    #endif

    if(_graph.get_container_type() == VECTOR_CSR_GRAPH)
    {
        VectorCSRGraph *container_graph = (VectorCSRGraph *)_graph.get_incoming_data();
        FrontierVectorCSR *container_frontier = (FrontierVectorCSR *)_frontier.get_container_data();

        #ifdef __USE_GPU__
        _abstraction_class->advance_worker(*container_graph, *container_frontier, edge_op,
                                           vertex_preprocess_op, vertex_postprocess_op,
                                           collective_edge_op, collective_vertex_preprocess_op,
                                           collective_vertex_postprocess_op,
                                           inner_mpi_processing);
        #else
        OMP_PARALLEL_CALL((_abstraction_class->advance_worker(*container_graph, *container_frontier, edge_op,
                                                              vertex_preprocess_op, vertex_postprocess_op,
                                                              collective_edge_op, collective_vertex_preprocess_op,
                                                              collective_vertex_postprocess_op,
                                                              inner_mpi_processing)));
        #endif
    }
    else if(_graph.get_container_type() == EDGES_LIST_GRAPH)
    {
        EdgesListGraph *container_graph = (EdgesListGraph *)_graph.get_incoming_data();
        FrontierEdgesList *container_frontier = (FrontierEdgesList *)_frontier.get_container_data();

        #ifdef __USE_GPU__
        _abstraction_class->advance_worker(*container_graph, *container_frontier, edge_op,
                                           vertex_preprocess_op, vertex_postprocess_op,
                                           collective_edge_op, collective_vertex_preprocess_op,
                                           collective_vertex_postprocess_op,
                                           inner_mpi_processing);
        #else
        OMP_PARALLEL_CALL((_abstraction_class->advance_worker(*container_graph, *container_frontier, edge_op,
                                                              vertex_preprocess_op, vertex_postprocess_op,
                                                              collective_edge_op, collective_vertex_preprocess_op,
                                                              collective_vertex_postprocess_op,
                                                              inner_mpi_processing)));
        #endif
    }
    else if(_graph.get_container_type() == CSR_GRAPH)
    {
        CSRGraph *container_graph = (CSRGraph *)_graph.get_incoming_data();
        FrontierCSR *container_frontier = (FrontierCSR *)_frontier.get_container_data();

        #ifdef __USE_GPU__
        _abstraction_class->advance_worker(*container_graph, *container_frontier, edge_op,
                                           vertex_preprocess_op, vertex_postprocess_op,
                                           collective_edge_op, collective_vertex_preprocess_op,
                                           collective_vertex_postprocess_op,
                                           inner_mpi_processing);
        #else
        OMP_PARALLEL_CALL((_abstraction_class->advance_worker(*container_graph, *container_frontier, edge_op,
                                                              vertex_preprocess_op, vertex_postprocess_op,
                                                              collective_edge_op, collective_vertex_preprocess_op,
                                                              collective_vertex_postprocess_op,
                                                              inner_mpi_processing)));
        #endif
    }
    else if(_graph.get_container_type() == CSR_VG_GRAPH)
    {
        CSR_VG_Graph *container_graph = (CSR_VG_Graph *)_graph.get_incoming_data();
        FrontierCSR_VG *container_frontier = (FrontierCSR_VG *)_frontier.get_container_data();

        #ifdef __USE_GPU__
        _abstraction_class->advance_worker(*container_graph, *container_frontier, edge_op,
                                           vertex_preprocess_op, vertex_postprocess_op,
                                           collective_edge_op, collective_vertex_preprocess_op,
                                           collective_vertex_postprocess_op,
                                           inner_mpi_processing);
        #else
        OMP_PARALLEL_CALL((_abstraction_class->advance_worker(*container_graph, *container_frontier, edge_op,
                                                              vertex_preprocess_op, vertex_postprocess_op,
                                                              collective_edge_op, collective_vertex_preprocess_op,
                                                              collective_vertex_postprocess_op,
                                                              inner_mpi_processing)));
        #endif
    }
    else
    {
        throw "Error in GraphAbstractions::gather unsupported graph type";
    }

    tm.end();
    performance_stats.update_scatter_time(tm);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
