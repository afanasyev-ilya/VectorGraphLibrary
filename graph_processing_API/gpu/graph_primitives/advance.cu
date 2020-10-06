#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphPrimitivesGPU::advance(ExtendedCSRGraph &_graph,
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
void GraphPrimitivesGPU::advance(ExtendedCSRGraph &_graph,
                                 FrontierGPU &_frontier,
                                 EdgeOperation edge_op)
{
    auto EMPTY_VERTEX_OP = [] __device__(int src_id, int position_in_frontier, int connections_count){};
    advance_sparse(_graph, _frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename Condition>
void GraphPrimitivesGPU::advance(ExtendedCSRGraph &_graph,
                                 FrontierGPU &_in_frontier,
                                 EdgeOperation edge_op,
                                 VertexPreprocessOperation vertex_preprocess_op,
                                 VertexPostprocessOperation vertex_postprocess_op,
                                 FrontierGPU &_out_frontier,
                                 Condition &&cond)
{
    int vertices_count = _graph.get_vertices_count();
    int *adjacent_layer_size;
    MemoryAPI::allocate_managed_array(&adjacent_layer_size, 1);
    bool generate_frontier_inside_advance = false;

    int in_frontier_size = _in_frontier.size();
    if(double(in_frontier_size) / vertices_count < 0.005)
    {
        auto reduce_op = [] __device__ (int src_id, int connections_count)->int
        {
            return connections_count;
        };
        adjacent_layer_size[0] = reduce<int>(_graph, _in_frontier, reduce_op, REDUCE_SUM);
        cout << "adjacent_layer_size[0]: " << adjacent_layer_size[0] << endl;

        if(double(adjacent_layer_size[0]) / vertices_count < 0.001)
            generate_frontier_inside_advance = true;
    }

    cout << "generate_frontier_inside_advance: " << generate_frontier_inside_advance << endl;

    if(generate_frontier_inside_advance)
    {
        //advance_sparse(_graph, _in_frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op);
        //generate_new_frontier(_graph, _out_frontier, cond);
        advance_sparse(_graph, _in_frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op);
        generate_new_frontier(_graph, _out_frontier, cond);
    }
    else
    {
        advance_sparse(_graph, _in_frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op);
        generate_new_frontier(_graph, _out_frontier, cond);
    }


    //MemoryAPI::free_array(adjacent_layer_size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
