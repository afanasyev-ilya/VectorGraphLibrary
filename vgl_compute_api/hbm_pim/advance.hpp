#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation, typename GraphContainer, typename FrontierContainer>
void GraphAbstractionsTEMPLATE::advance_worker(GraphContainer &_graph,
                                               FrontierContainer &_frontier,
                                               EdgeOperation &&edge_op,
                                               VertexPreprocessOperation &&vertex_preprocess_op,
                                               VertexPostprocessOperation &&vertex_postprocess_op,
                                               CollectiveEdgeOperation &&collective_edge_op,
                                               CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                                               CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op,
                                               bool _inner_mpi_processing)
{
    throw "Error in GraphAbstractionsTEMPLATE::advance_worker : not implemented yet";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
