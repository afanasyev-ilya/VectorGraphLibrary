#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation>
void GraphAbstractionsMulticore::gather(VectCSRGraph &_graph,
                                        FrontierMulticore &_frontier,
                                        EdgeOperation &&edge_op,
                                        VertexPreprocessOperation &&vertex_preprocess_op,
                                        VertexPostprocessOperation &&vertex_postprocess_op,
                                        CollectiveEdgeOperation &&collective_edge_op,
                                        CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                                        CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op)
{
    Timer tm;
    tm.start();
    UndirectedCSRGraph *current_direction_graph;

    if(current_traversal_direction != GATHER)
    {
        throw "Error in GraphAbstractionsNEC::gather : wrong traversal direction";
    }
    if(_frontier.get_direction() != current_traversal_direction)
    {
        throw "Error in GraphAbstractionsNEC::gather : wrong frontier direction";
    }
    current_direction_graph = _graph.get_incoming_graph_ptr();

    bool outgoing_graph_is_stored = _graph.outgoing_is_stored();

    if(omp_in_parallel())
    {
        #pragma omp barrier
        advance_worker(*current_direction_graph, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op,
                       collective_edge_op, collective_vertex_preprocess_op, collective_vertex_postprocess_op, 0, 0,
                       outgoing_graph_is_stored);
        #pragma omp barrier
    }
    else
    {
        #pragma omp parallel
        {
            advance_worker(*current_direction_graph, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op,
                           collective_edge_op, collective_vertex_preprocess_op, collective_vertex_postprocess_op, 0, 0,
                           outgoing_graph_is_stored);
        }
    }

    tm.end();
    performance_stats.update_gather_time(tm);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation>
void GraphAbstractionsMulticore::gather(VectCSRGraph &_graph,
                                        FrontierMulticore &_frontier,
                                        EdgeOperation &&edge_op)
{
    gather(_graph, _frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
           edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
