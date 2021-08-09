#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphAbstractions
{
protected:
    vector<UserDataContainer*> user_data_containers;

    VGL_Graph *processed_graph_ptr;
    TraversalDirection current_traversal_direction;

    inline size_t compute_process_shift(TraversalDirection _traversal, int _storage);

    bool same_direction(TraversalDirection _first, TraversalDirection _second);

    // allows to set correct direction for multiple arrays (vertexArrays, frontiers)
    void set_correct_direction();
    template<typename _T, typename ... Types>
    void set_correct_direction(_T &_first_arg, Types &... _args);
public:
    // attaches graph-processing API to the specific graph
    GraphAbstractions();

    // performs user-defined "edge_op" operation over all OUTGOING edges, neighbouring specified frontier
    template <typename EdgeOperation, typename VertexPreprocessOperation, typename VertexPostprocessOperation,
            typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation>
    void scatter(VGL_Graph &_graph,
                 VGL_Frontier &_frontier,
                 EdgeOperation &&edge_op,
                 VertexPreprocessOperation &&vertex_preprocess_op,
                 VertexPostprocessOperation &&vertex_postprocess_op,
                 CollectiveEdgeOperation &&collective_edge_op,
                 CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                 CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op);

    // performs user-defined "edge_op" operation over all INCOMING edges, neighbouring specified frontier
    template <typename EdgeOperation, typename VertexPreprocessOperation, typename VertexPostprocessOperation,
            typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation>
    void gather(VGL_Graph &_graph,
                VGL_Frontier &_frontier,
                EdgeOperation &&edge_op,
                VertexPreprocessOperation &&vertex_preprocess_op,
                VertexPostprocessOperation &&vertex_postprocess_op,
                CollectiveEdgeOperation &&collective_edge_op,
                CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op);

    // performs user-defined "compute_op" operation for each element in the given frontier
    template <typename ComputeOperation>
    void compute(VGL_Graph &_graph,
                 VGL_Frontier &_frontier,
                 ComputeOperation &&compute_op);

    // performs reduction using user-defined "reduce_op" operation for each element in the given frontier
    template <typename _T, typename ReduceOperation>
    _T reduce(VGL_Graph &_graph,
              VGL_Frontier &_frontier,
              ReduceOperation &&reduce_op,
              REDUCE_TYPE _reduce_type);

    // creates new frontier, which satisfy user-defined "cond" condition
    template <typename FilterCondition>
    void generate_new_frontier(VGL_Graph &_graph,
                               VGL_Frontier &_frontier,
                               FilterCondition &&filter_cond);

    // allows to check if multiple arrays (vertexArrays, frontiers) have correct direction
    bool have_correct_direction();
    template<typename _T, typename ... Types>
    bool have_correct_direction(_T _first_arg, Types ... _args);

    // change graph traversal direction (from GATHER to SCATTER or vice versa)
    template<typename _T, typename ... Types>
    void change_traversal_direction(TraversalDirection _new_direction, _T &_first_arg, Types &... _args);
    void change_traversal_direction(TraversalDirection _new_direction);

    template<typename _T>
    void attach_data(VerticesArray<_T> &_array);

    #ifdef __USE_MPI__
    template <typename _TGraph, typename _T>
    void exchange_vertices_array(DataExchangePolicy _policy, _TGraph &_graph, VerticesArray<_T> &_data);

    template <typename _TGraph, typename _T, typename MergeOp>
    void exchange_vertices_array(DataExchangePolicy _policy, _TGraph &_graph, VerticesArray<_T> &_data,
                                 MergeOp &&_merge_op);

    template <typename _TGraph, typename _T, typename MergeOp>
    void exchange_vertices_array(DataExchangePolicy _policy, _TGraph &_graph, VerticesArray<_T> &_data,
                                 VerticesArray<_T> &_old_data, MergeOp &&_merge_op);
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_abstractions.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
#include "vgl_compute_api/nec/graph_abstractions_nec.h"
#endif

#ifdef __USE_GPU__
#include "vgl_compute_api/gpu/graph_abstractions_gpu.cuh"
#endif

#if defined(__USE_MULTICORE__)
#include "vgl_compute_api/multicore/graph_abstractions_multicore.h"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
