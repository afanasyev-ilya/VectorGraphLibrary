#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphAbstractions
{
protected:
    VectCSRGraph *processed_graph_ptr;
    TraversalDirection current_traversal_direction;

    bool same_direction(TraversalDirection _first, TraversalDirection _second) {return (_first == _second);};
public:
    // attaches graph-processing API to the specific graph
    GraphAbstractionsNEC() {};

    // change graph traversal direction (from GATHER to SCATTER or vice versa)
    void change_traversal_direction(TraversalDirection _new_direction);

    // performs user-defined "edge_op" operation over all OUTGOING edges, neighbouring specified frontier
    template <typename EdgeOperation, typename VertexPreprocessOperation, typename VertexPostprocessOperation,
            typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation>
    void scatter(VectCSRGraph &_graph,
                 FrontierNEC &_frontier,
                 EdgeOperation &&edge_op,
                 VertexPreprocessOperation &&vertex_preprocess_op,
                 VertexPostprocessOperation &&vertex_postprocess_op,
                 CollectiveEdgeOperation &&collective_edge_op,
                 CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                 CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op) {};

    // performs user-defined "edge_op" operation over all INCOMING edges, neighbouring specified frontier
    template <typename EdgeOperation, typename VertexPreprocessOperation, typename VertexPostprocessOperation,
            typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation>
    void gather(VectCSRGraph &_graph,
                FrontierNEC &_frontier,
                EdgeOperation &&edge_op,
                VertexPreprocessOperation &&vertex_preprocess_op,
                VertexPostprocessOperation &&vertex_postprocess_op,
                CollectiveEdgeOperation &&collective_edge_op,
                CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op) {};

    // performs user-defined "compute_op" operation for each element in the given frontier
    template <typename ComputeOperation>
    void compute(VectCSRGraph &_graph,
                 FrontierNEC &_frontier,
                 ComputeOperation &&compute_op) {};

    // performs reduction using user-defined "reduce_op" operation for each element in the given frontier
    template <typename _T, typename ReduceOperation>
    _T reduce(VectCSRGraph &_graph,
              FrontierNEC &_frontier,
              ReduceOperation &&reduce_op,
              REDUCE_TYPE _reduce_type) {return 0;};

    // creates new frontier, which satisfy user-defined "cond" condition
    template <typename FilterCondition>
    void generate_new_frontier(VectCSRGraph &_graph,
                               FrontierNEC &_frontier,
                               FilterCondition &&filter_cond) {};

    // allows to check if multiple arrays (vertexArrays, frontiers) have correct direction
    bool have_correct_direction();
    template<typename _T, typename ... Types>
    bool have_correct_direction(_T _first_arg, Types ... _args);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GraphAbstractions::change_traversal_direction(TraversalDirection _new_direction)
{
    current_traversal_direction = _new_direction;
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

#include "nec/graph_abstractions/graph_abstractions_nec.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
