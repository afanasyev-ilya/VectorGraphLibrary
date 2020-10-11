#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
void SCC::bfs_reach(VectCSRGraph &_graph,
                    GraphAbstractionsNEC &_graph_API,
                    FrontierNEC &frontier,
                    VerticesArrayNec<int> &_bfs_result,
                    int _source_vertex)
{
    _graph_API.change_traversal_direction(SCATTER);
    _frontier.set_all_active();


}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
void SCC::nec_forward_backward(VectCSRGraph &_graph, VerticesArrayNec<int> &_components)
{
    cout << "nec FB" << endl;

    GraphAbstractionsNEC graph_API(_graph, SCATTER);
    FrontierNEC frontier(_graph, SCATTER);

    VerticesArrayNec<int> &forward_result(_graph, SCATTER);
    VerticesArrayNec<int> &backward_result(_graph, GATHER);

    int pivot = 0; //select_pivot(_trees, _tree_num);
    //int pivot_in_reversed = f(pivot)

    /*bfs_reach(src_ids, dst_ids, pivot, fwd_result, _trees);
    bfs_reach(dst_ids, src_ids, pivot, bwd_result, _trees);

    int loc_last_trees[3] = { 0, 0, 0 };
    process_result(fwd_result, bwd_result, _components, _trees, _active, _last_component, loc_last_trees);*/
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

