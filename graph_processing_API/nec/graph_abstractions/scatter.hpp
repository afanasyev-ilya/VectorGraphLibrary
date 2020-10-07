#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation>
void GraphAbstractionsNEC::scatter(VectCSRGraph &_graph,
                                   FrontierNEC &_frontier,
                                   EdgeOperation &&edge_op,
                                   VertexPreprocessOperation &&vertex_preprocess_op,
                                   VertexPostprocessOperation &&vertex_postprocess_op,
                                   CollectiveEdgeOperation &&collective_edge_op,
                                   CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                                   CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op)
{
    ExtendedCSRGraph *current_direction_graph;

    // TODO is frontier direction correct?

    if(traversal_direction == SCATTER_TRAVERSAL)
    {
        current_direction_graph = _graph.get_outgoing_graph_ptr();
    }
    else if(traversal_direction == GATHER_TRAVERSAL)
    {
        current_direction_graph = _graph.get_incoming_graph_ptr();
    }

    if(omp_in_parallel())
    {
        #pragma omp barrier
        advance_worker(*current_direction_graph, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op,
                       collective_edge_op, collective_vertex_preprocess_op, collective_vertex_postprocess_op, 0);
        #pragma omp barrier
    }
    else
    {
        #pragma omp parallel
        {
            advance_worker(*current_direction_graph, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op,
                           collective_edge_op, collective_vertex_preprocess_op, collective_vertex_postprocess_op, 0);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
