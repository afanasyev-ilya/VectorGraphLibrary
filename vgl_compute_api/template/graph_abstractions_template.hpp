#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GraphAbstractionsTEMPLATE::GraphAbstractionsTEMPLATE(VGL_Graph &_graph, TraversalDirection _initial_traversal)
{
    processed_graph_ptr = &_graph;
    current_traversal_direction = _initial_traversal;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GraphAbstractionsTEMPLATE::~GraphAbstractionsTEMPLATE()
{

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation>
void GraphAbstractionsTEMPLATE::gather(VGL_Graph &_graph,
                                  VGL_Frontier &_frontier,
                                  EdgeOperation &&edge_op,
                                  VertexPreprocessOperation &&vertex_preprocess_op,
                                  VertexPostprocessOperation &&vertex_postprocess_op,
                                  CollectiveEdgeOperation &&collective_edge_op,
                                  CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                                  CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op)
{
    this->common_gather(_graph, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op,
                        collective_edge_op, collective_vertex_preprocess_op, collective_vertex_postprocess_op, this);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation>
void GraphAbstractionsTEMPLATE::gather(VGL_Graph &_graph,
                                  VGL_Frontier &_frontier,
                                  EdgeOperation &&edge_op)
{
    gather(_graph, _frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
           edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation>
void GraphAbstractionsTEMPLATE::scatter(VGL_Graph &_graph,
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
void GraphAbstractionsTEMPLATE::scatter(VGL_Graph &_graph,
                                   VGL_Frontier &_frontier,
                                   EdgeOperation &&edge_op)
{
    scatter(_graph, _frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
            edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ComputeOperation>
void GraphAbstractionsTEMPLATE::compute(VGL_Graph &_graph,
                                   VGL_Frontier &_frontier,
                                   ComputeOperation &&compute_op)
{
    this->common_compute(_graph, _frontier, compute_op, this);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename ReduceOperation>
_T GraphAbstractionsTEMPLATE::reduce(VGL_Graph &_graph,
                                VGL_Frontier &_frontier,
                                ReduceOperation &&reduce_op,
                                REDUCE_TYPE _reduce_type)
{
    _T result = 0;
    this->common_reduce(_graph, _frontier, reduce_op, _reduce_type, result, this);
    return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



