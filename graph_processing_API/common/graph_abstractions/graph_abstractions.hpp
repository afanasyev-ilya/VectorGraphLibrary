#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool GraphAbstractions::same_direction(TraversalDirection _first, TraversalDirection _second)
{
    return (_first == _second);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GraphAbstractions::GraphAbstractions()
{

}

/*
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation, typename VertexPostprocessOperation,
        typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation>
void GraphAbstractions::scatter(VectCSRGraph &_graph,
                                FrontierNEC &_frontier,
                                EdgeOperation &&edge_op,
                                VertexPreprocessOperation &&vertex_preprocess_op,
                                VertexPostprocessOperation &&vertex_postprocess_op,
                                CollectiveEdgeOperation &&collective_edge_op,
                                CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                                CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op)
{

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation, typename VertexPostprocessOperation,
        typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation>
void GraphAbstractions::gather(VectCSRGraph &_graph,
                               FrontierNEC &_frontier,
                               EdgeOperation &&edge_op,
                               VertexPreprocessOperation &&vertex_preprocess_op,
                               VertexPostprocessOperation &&vertex_postprocess_op,
                               CollectiveEdgeOperation &&collective_edge_op,
                               CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                               CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op)
{

}*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ComputeOperation>
void GraphAbstractions::compute(VGL_Graph &_graph,
                                VGL_Frontier &_frontier,
                                ComputeOperation &&compute_op)
{

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
template <typename _T, typename ReduceOperation>
_T GraphAbstractions::reduce(VectCSRGraph &_graph,
                             FrontierNEC &_frontier,
                             ReduceOperation &&reduce_op,
                             REDUCE_TYPE _reduce_type)
{
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FilterCondition>
void GraphAbstractions::generate_new_frontier(VectCSRGraph &_graph,
                                              FrontierNEC &_frontier,
                                              FilterCondition &&filter_cond)
{

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void GraphAbstractions::change_traversal_direction(TraversalDirection _new_direction)
{
    current_traversal_direction = _new_direction;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename _T, typename ... Types>
void GraphAbstractions::change_traversal_direction(TraversalDirection _new_direction, _T &_first_arg, Types &... _args)
{
    current_traversal_direction = _new_direction;
    set_correct_direction(_first_arg, _args...);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool GraphAbstractions::have_correct_direction()
{
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename _T, typename ... _Types>
bool GraphAbstractions::have_correct_direction(_T _first_arg, _Types ... _args)
{
    bool check_result = same_direction(_first_arg.get_direction(), current_traversal_direction);
    return (check_result && have_correct_direction(_args...));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GraphAbstractions::set_correct_direction()
{

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename _T, typename ... Types>
void GraphAbstractions::set_correct_direction(_T &_first_arg, Types &... _args)
{
    _first_arg.reorder(current_traversal_direction);

    set_correct_direction(_args...);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename _T>
void GraphAbstractions::attach_data(VerticesArray<_T> &_array)
{
    user_data_containers.push_back(&_array);
}
*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



