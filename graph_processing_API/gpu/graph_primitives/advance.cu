#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphPrimitivesGPU::advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                 FrontierGPU &_frontier,
                                 EdgeOperation edge_op,
                                 VertexPreprocessOperation vertex_preprocess_op,
                                 VertexPostprocessOperation vertex_postprocess_op)
{
    if(_frontier.type == SPARSE_FRONTIER || _frontier.type == ALL_ACTIVE_FRONTIER || _frontier.type == DENSE_FRONTIER)
    {
        advance_sparse(_graph, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation>
void GraphPrimitivesGPU::advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                 FrontierGPU &_frontier,
                                 EdgeOperation edge_op)
{
    auto EMPTY_VERTEX_OP = [] __device__(int src_id, int position_in_frontier, int connections_count){};

    advance_sparse(_graph, _frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
