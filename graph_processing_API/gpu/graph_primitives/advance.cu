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
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    double t1 = omp_get_wtime();
    #endif

    if(_frontier.type == SPARSE_FRONTIER || _frontier.type == ALL_ACTIVE_FRONTIER || _frontier.type == DENSE_FRONTIER)
    {
        advance_sparse(_graph, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op);
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    double t2 = omp_get_wtime();
    double max_work = _graph.get_edges_count();
    auto compute_work_size = []__device__(int src_id, int position_in_frontier, int connections_count)->int
    {
        return connections_count;
    };
    double work = this->reduce<int>(_graph, _frontier, compute_work_size, REDUCE_SUM);
    cout << "real work: " << work << " vs max work: " << max_work << endl;
    cout << "advance time: " << (t2 - t1)*1000.0 << " ms" << endl;
    cout << "advance sparse BW: " << sizeof(int)*INT_ELEMENTS_PER_EDGE*work/((t2-t1)*1e9) << " GB/s" << endl << endl;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation>
void GraphPrimitivesGPU::advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                 FrontierGPU &_frontier,
                                 EdgeOperation edge_op)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    double t1 = omp_get_wtime();
    #endif

    auto EMPTY_VERTEX_OP = [] __device__(int src_id, int position_in_frontier, int connections_count){};
    advance_sparse(_graph, _frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    double t2 = omp_get_wtime();
    double max_work = _graph.get_edges_count();
    auto compute_work_size = []__device__(int src_id, int position_in_frontier, int connections_count)->int
    {
        return connections_count;
    };
    double work = this->reduce<int>(_graph, _frontier, compute_work_size, REDUCE_SUM);
    cout << "real work: " << work << " vs max work: " << max_work << endl;
    cout << "advance time: " << (t2 - t1)*1000.0 << " ms" << endl;
    cout << "advance sparse BW: " << sizeof(int)*INT_ELEMENTS_PER_EDGE*work/((t2-t1)*1e9) << " GB/s" << endl << endl;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename Condition>
void GraphPrimitivesGPU::advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
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
