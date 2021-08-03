#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation>
void GraphAbstractionsMulticore::gather(VGL_Graph &_graph,
                                        VGL_Frontier &_frontier,
                                        EdgeOperation &&edge_op,
                                        VertexPreprocessOperation &&vertex_preprocess_op,
                                        VertexPostprocessOperation &&vertex_postprocess_op,
                                        CollectiveEdgeOperation &&collective_edge_op,
                                        CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                                        CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op)
{
    Timer tm;
    tm.start();

    if(current_traversal_direction != GATHER)
    {
        throw "Error in GraphAbstractionsMulticore::gather : wrong traversal direction";
    }
    if(_frontier.get_direction() != current_traversal_direction)
    {
        throw "Error in GraphAbstractionsMulticore::gather : wrong frontier direction";
    }
    UndirectedGraph *current_direction_graph = _graph.get_incoming_data();

    bool outgoing_graph_is_stored = _graph.outgoing_is_stored();
    if(_graph.get_container_type() == VECTOR_CSR_GRAPH)
    {
        VectorCSRGraph *current_direction_graph = (VectorCSRGraph *)_graph.get_incoming_data();
        FrontierVectorCSR *current_frontier = (FrontierVectorCSR *)_frontier.get_container_data();
        if(omp_in_parallel())
        {
            #pragma omp barrier
            advance_worker(*current_direction_graph, *current_frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op,
                           collective_edge_op, collective_vertex_preprocess_op, collective_vertex_postprocess_op, 0, 0,
                           outgoing_graph_is_stored);
            #pragma omp barrier
        }
        else
        {
            #pragma omp parallel
            {
                advance_worker(*current_direction_graph, *current_frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op,
                               collective_edge_op, collective_vertex_preprocess_op, collective_vertex_postprocess_op, 0, 0,
                               outgoing_graph_is_stored);
            }
        }
    }
    else
    {
        throw "Error in GraphAbstractionsMulticore::gather unsupported graph type";
    }

    tm.end();
    performance_stats.update_gather_time(tm);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation>
void GraphAbstractionsMulticore::gather(VGL_Graph &_graph,
                                        VGL_Frontier &_frontier,
                                        EdgeOperation &&edge_op)
{
    gather(_graph, _frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
           edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
