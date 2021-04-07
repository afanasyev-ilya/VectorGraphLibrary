#pragma once

#include <set>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation>
void GraphAbstractionsMulticore::advance_worker(EdgesListGraph &_graph,
                                          EdgeOperation &&edge_op)
{
    Timer tm;
    tm.start();
    LOAD_EDGES_LIST_GRAPH_DATA(_graph);

    DelayedWriteNEC delayed_write;
    delayed_write.init();

    #pragma omp for schedule(static)
    for(long long vec_start = 0; vec_start < edges_count; vec_start += VECTOR_LENGTH)
    {
        #pragma simd
        #pragma vector
        #pragma ivdep
        #pragma unroll(VECTOR_LENGTH)
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            long long edge_pos = vec_start + i;
            if((vec_start + i) < edges_count)
            {
                int src_id = src_ids[edge_pos];
                int dst_id = dst_ids[edge_pos];
                int vector_index = i;
                edge_op(src_id, dst_id, edge_pos, edge_pos, vector_index, delayed_write);
            }
        }
    }

    tm.end();
    performance_stats.update_advance_time(tm);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation>
void GraphAbstractionsMulticore::advance_worker(UndirectedCSRGraph &_graph,
                                          FrontierMulticore &_frontier,
                                          EdgeOperation &&edge_op,
                                          VertexPreprocessOperation &&vertex_preprocess_op,
                                          VertexPostprocessOperation &&vertex_postprocess_op,
                                          CollectiveEdgeOperation &&collective_edge_op,
                                          CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                                          CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op,
                                          int _first_edge,
                                          const long long _shard_shift,
                                          bool _outgoing_graph_is_stored)
{
    Timer tm;
    tm.start();

    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);
    const int vector_engine_threshold_start = 0;
    const int vector_engine_threshold_end = _graph.get_vector_engine_threshold_vertex();
    const int vector_core_threshold_start = _graph.get_vector_engine_threshold_vertex();
    const int vector_core_threshold_end = _graph.get_vector_core_threshold_vertex();
    const int collective_threshold_start = _graph.get_vector_core_threshold_vertex();
    const int collective_threshold_end = _graph.get_vertices_count();

    if(_frontier.type == ALL_ACTIVE_FRONTIER)
    {
        if((vector_engine_threshold_end - vector_engine_threshold_start) > 0)
            vector_engine_per_vertex_kernel_all_active(_graph, vector_engine_threshold_start,
                                                       vector_engine_threshold_end, edge_op, vertex_preprocess_op,
                                                       vertex_postprocess_op, _first_edge, _shard_shift,
                                                       _outgoing_graph_is_stored);

        if((vector_core_threshold_end - vector_core_threshold_start) > 0)
            vector_core_per_vertex_kernel_all_active(_graph, vector_core_threshold_start,
                                                     vector_core_threshold_end, edge_op, vertex_preprocess_op,
                                                     vertex_postprocess_op, _first_edge, _shard_shift,
                                                     _outgoing_graph_is_stored);

        if((collective_threshold_end - collective_threshold_start) > 0)
            ve_collective_vertex_processing_kernel_all_active(_graph, collective_threshold_start, collective_threshold_end,
                                                              collective_edge_op, collective_vertex_preprocess_op,
                                                              collective_vertex_postprocess_op, _first_edge, _shard_shift,
                                                              _outgoing_graph_is_stored);
    }
    else
    {
        if(_frontier.vector_engine_part_size > 0)
        {
            if (_frontier.vector_engine_part_type == DENSE_FRONTIER)
            {
                vector_engine_per_vertex_kernel_dense(_graph, _frontier, vector_engine_threshold_start, vector_engine_threshold_end,
                                                      edge_op, vertex_preprocess_op, vertex_postprocess_op,
                                                      _first_edge, _outgoing_graph_is_stored);
            }
            else if (_frontier.vector_engine_part_type == SPARSE_FRONTIER)
            {
                vector_engine_per_vertex_kernel_sparse(_graph, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op,
                                                       _first_edge, _outgoing_graph_is_stored);
            }
        }

        if(_frontier.vector_core_part_size > 0)
        {
            if(_frontier.vector_core_part_type == DENSE_FRONTIER)
            {
                vector_core_per_vertex_kernel_dense(_graph, _frontier, vector_core_threshold_start, vector_core_threshold_end, edge_op,
                                                    vertex_preprocess_op, vertex_postprocess_op, _first_edge, _outgoing_graph_is_stored);
            }
            else if(_frontier.vector_core_part_type == SPARSE_FRONTIER)
            {
                vector_core_per_vertex_kernel_sparse(_graph, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op, _first_edge,
                                                     _outgoing_graph_is_stored);
            }
        }

        if(_frontier.collective_part_size > 0)
        {
            if(_frontier.collective_part_type == DENSE_FRONTIER)
            {
                ve_collective_vertex_processing_kernel_dense(_graph, _frontier, collective_threshold_start, collective_threshold_end,
                                                             collective_edge_op, collective_vertex_preprocess_op,
                                                             collective_vertex_postprocess_op, _first_edge,
                                                             _outgoing_graph_is_stored);
            }
            else if(_frontier.collective_part_type == SPARSE_FRONTIER)
            {
                collective_vertex_processing_kernel_sparse(_graph, _frontier, collective_threshold_start,
                                                           collective_threshold_end, collective_edge_op,
                                                           collective_vertex_preprocess_op,
                                                           collective_vertex_postprocess_op, _first_edge,
                                                           _outgoing_graph_is_stored);
            }
        }
    }

    tm.end();
    performance_stats.update_advance_time(tm);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
