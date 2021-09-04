#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GraphAbstractionsGPU::GraphAbstractionsGPU(VGL_Graph &_graph, TraversalDirection _initial_traversal)
{
    processed_graph_ptr = &_graph;
    current_traversal_direction = _initial_traversal;

    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);
    cudaStreamCreate(&stream_3);
    cudaStreamCreate(&stream_4);
    cudaStreamCreate(&stream_5);
    cudaStreamCreate(&stream_6);

    MemoryAPI::allocate_array(&reduce_buffer, _graph.get_vertices_count());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GraphAbstractionsGPU::~GraphAbstractionsGPU()
{
    cudaStreamDestroy(stream_1);
    cudaStreamDestroy(stream_2);
    cudaStreamDestroy(stream_3);
    cudaStreamDestroy(stream_4);
    cudaStreamDestroy(stream_5);
    cudaStreamDestroy(stream_6);
    MemoryAPI::free_array(reduce_buffer);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation>
void GraphAbstractionsGPU::gather(VGL_Graph &_graph,
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
void GraphAbstractionsGPU::gather(VGL_Graph &_graph,
                                  VGL_Frontier &_frontier,
                                  EdgeOperation &&edge_op)
{
    auto EMPTY_VERTEX_OP = [] __device__(int src_id, int position_in_frontier, int connections_count){};
    gather(_graph, _frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
           edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation>
void GraphAbstractionsGPU::scatter(VGL_Graph &_graph,
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
void GraphAbstractionsGPU::scatter(VGL_Graph &_graph,
                                   VGL_Frontier &_frontier,
                                   EdgeOperation &&edge_op)
{
    auto EMPTY_VERTEX_OP = [] __device__(int src_id, int position_in_frontier, int connections_count){};
    scatter(_graph, _frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
            edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ComputeOperation>
void GraphAbstractionsGPU::compute(VGL_Graph &_graph,
                                   VGL_Frontier &_frontier,
                                   ComputeOperation &&compute_op)
{
    this->common_compute(_graph, _frontier, compute_op, this);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename ReduceOperation>
_T GraphAbstractionsGPU::reduce(VGL_Graph &_graph,
                                VGL_Frontier &_frontier,
                                ReduceOperation &&reduce_op,
                                REDUCE_TYPE _reduce_type)
{
    _T result = 0;
    this->common_reduce(_graph, _frontier, reduce_op, _reduce_type, result, this);
    return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



