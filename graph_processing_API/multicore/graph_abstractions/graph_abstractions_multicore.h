#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

auto EMPTY_EDGE_OP = [] (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int vector_index) {};
auto EMPTY_VERTEX_OP = [] (int src_id, int connections_count, int vector_index){};

auto ALL_ACTIVE_FRONTIER_CONDITION = [] (int src_id)->int
{
    return IN_FRONTIER_FLAG;
};

auto EMPTY_COMPUTE_OP = [] (int src_id, int connections_count, int vector_index) {};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphAbstractionsMulticore : public GraphAbstractions
{
private:
    template <typename EdgeOperation, typename VertexPreprocessOperation,
            typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation>
    inline void advance_worker(UndirectedCSRGraph &_graph,
                        FrontierMulticore &_frontier,
                        EdgeOperation &&edge_op,
                        VertexPreprocessOperation &&vertex_preprocess_op,
                        VertexPostprocessOperation &&vertex_postprocess_op,
                        CollectiveEdgeOperation &&collective_edge_op,
                        CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                        CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op,
                        int _first_edge,
                        const long long _shard_shift,
                        bool _outgoing_graph_is_stored);
public:
    GraphAbstractionsMulticore(VectCSRGraph &_graph) {};

    // performs user-defined "edge_op" operation over all OUTGOING edges, neighbouring specified frontier
    template <typename EdgeOperation, typename VertexPreprocessOperation, typename VertexPostprocessOperation,
            typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation>
    inline void scatter(VectCSRGraph &_graph,
                 FrontierMulticore &_frontier,
                 EdgeOperation &&edge_op,
                 VertexPreprocessOperation &&vertex_preprocess_op,
                 VertexPostprocessOperation &&vertex_postprocess_op,
                 CollectiveEdgeOperation &&collective_edge_op,
                 CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                 CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op);

    // performs user-defined "edge_op" operation over all OUTGOING edges, neighbouring specified frontier
    template <typename EdgeOperation>
    inline void scatter(VectCSRGraph &_graph,
                 FrontierMulticore &_frontier,
                 EdgeOperation &&edge_op);

    // performs user-defined "edge_op" operation over all INCOMING edges, neighbouring specified frontier
    template <typename EdgeOperation, typename VertexPreprocessOperation, typename VertexPostprocessOperation,
            typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
            typename CollectiveVertexPostprocessOperation>
    inline void gather(VectCSRGraph &_graph,
                FrontierMulticore &_frontier,
                EdgeOperation &&edge_op,
                VertexPreprocessOperation &&vertex_preprocess_op,
                VertexPostprocessOperation &&vertex_postprocess_op,
                CollectiveEdgeOperation &&collective_edge_op,
                CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op);

    // performs user-defined "edge_op" operation over all INCOMING edges, neighbouring specified frontier
    template <typename EdgeOperation>
    inline void gather(VectCSRGraph &_graph,
                FrontierMulticore &_frontier,
                EdgeOperation &&edge_op);

    // performs user-defined "compute_op" operation for each element in the given frontier
    template <typename ComputeOperation>
    inline void compute(VectCSRGraph &_graph, FrontierMulticore &_frontier, ComputeOperation compute_op);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_abstractions_multicore.hpp"
#include "compute.hpp"
#include "advance.hpp"
#include "scatter.hpp"
#include "gather.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
