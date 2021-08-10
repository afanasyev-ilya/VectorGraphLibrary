#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation>
void GraphAbstractionsNEC::scatter(VGL_Graph &_graph,
                                   VGL_Frontier &_frontier,
                                   EdgeOperation &&edge_op,
                                   VertexPreprocessOperation &&vertex_preprocess_op,
                                   VertexPostprocessOperation &&vertex_postprocess_op,
                                   CollectiveEdgeOperation &&collective_edge_op,
                                   CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                                   CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op)
{
    this->common_scatter(_graph, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op,
                         collective_edge_op, collective_vertex_preprocess_op, collective_vertex_postprocess_op, this);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation>
void GraphAbstractionsNEC::scatter(VGL_Graph &_graph,
                                   VGL_Frontier &_frontier,
                                   EdgeOperation &&edge_op)
{
    scatter(_graph, _frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
            edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
