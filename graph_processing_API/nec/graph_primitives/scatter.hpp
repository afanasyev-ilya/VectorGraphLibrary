#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation>
void GraphPrimitivesNEC::scatter(VectCSRGraph &_graph,
                                 FrontierNEC &_frontier,
                                 EdgeOperation &&edge_op,
                                 VertexPreprocessOperation &&vertex_preprocess_op,
                                 VertexPostprocessOperation &&vertex_postprocess_op,
                                 CollectiveEdgeOperation &&collective_edge_op,
                                 CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                                 CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op)
{
    if(omp_in_parallel())
    {
        #pragma omp barrier
        //advance_worker(_graph, _frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP, edge_op, EMPTY_VERTEX_OP,
        //               EMPTY_VERTEX_OP);
        #pragma omp barrier
    }
    else
    {
        #pragma omp parallel
        {
        //    advance_worker(_graph, _frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP, edge_op, EMPTY_VERTEX_OP,
        //                   EMPTY_VERTEX_OP);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
