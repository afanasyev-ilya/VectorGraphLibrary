#pragma once

#include <set>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation>
void GraphAbstractionsNEC::advance_worker(EdgesListGraph &_graph,
                                          EdgeOperation &&edge_op)
{
    Timer tm;
    tm.start();
    LOAD_EDGES_LIST_GRAPH_DATA(_graph);

    #pragma omp for schedule(static)
    for(long long vec_start = 0; vec_start < edges_count; vec_start += VECTOR_LENGTH)
    {
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma _NEC gather_reorder
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            long long edge_pos = vec_start + i;
            if((vec_start + i) < edges_count)
            {
                int src_id = src_ids[edge_pos];
                int dst_id = dst_ids[edge_pos];
                int vector_index = i;
                edge_op(src_id, dst_id, edge_pos, edge_pos, vector_index);
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
void GraphAbstractionsNEC::advance_worker(UndirectedCSRGraph &_graph,
                                          FrontierNEC &_frontier,
                                          EdgeOperation &&edge_op,
                                          VertexPreprocessOperation &&vertex_preprocess_op,
                                          VertexPostprocessOperation &&vertex_postprocess_op,
                                          CollectiveEdgeOperation &&collective_edge_op,
                                          CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                                          CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op,
                                          int _first_edge,
                                          const long long _shard_shift,
                                          bool _outgoing_graph_is_stored,
                                          bool _inner_mpi_processing)
{
    double wall_time = 0, ve_time = 0, vc_time = 0, collective_time = 0, t1 = 0, t2 = 0;

    int vector_engine_threshold_start = 0;
    int vector_engine_threshold_end = 0;

    int vector_core_threshold_start = 0;
    int vector_core_threshold_end = 0;

    int collective_threshold_start = 0;
    int collective_threshold_end = 0;
    if(_inner_mpi_processing)
    {
        #ifdef __USE_MPI__
        pair<int,int> ve_mpi_borders = _frontier.get_vector_engine_mpi_thresholds();
        pair<int,int> vc_mpi_borders = _frontier.get_vector_core_mpi_thresholds();
        pair<int,int> coll_mpi_borders = _frontier.get_collective_mpi_thresholds();

        vector_engine_threshold_start = ve_mpi_borders.first;
        vector_engine_threshold_end = ve_mpi_borders.second;

        vector_core_threshold_start = vc_mpi_borders.first;
        vector_core_threshold_end = vc_mpi_borders.second;

        collective_threshold_start = coll_mpi_borders.first;
        collective_threshold_end = coll_mpi_borders.second;
        #endif
    }
    else
    {
        vector_engine_threshold_start = 0;
        vector_engine_threshold_end = _graph.get_vector_engine_threshold_vertex();
        vector_core_threshold_start = _graph.get_vector_engine_threshold_vertex();
        vector_core_threshold_end = _graph.get_vector_core_threshold_vertex();
        collective_threshold_start = _graph.get_vector_core_threshold_vertex();
        collective_threshold_end = _graph.get_vertices_count();
    }

    if(_frontier.type == ALL_ACTIVE_FRONTIER)
    {
        t1 = omp_get_wtime();
        if((vector_engine_threshold_end - vector_engine_threshold_start) > 0)
            vector_engine_per_vertex_kernel_all_active(_graph, vector_engine_threshold_start,
                                                       vector_engine_threshold_end, edge_op, vertex_preprocess_op,
                                                       vertex_postprocess_op, _first_edge, _shard_shift,
                                                       _outgoing_graph_is_stored);
        t2 = omp_get_wtime();
        ve_time += t2 - t1;

        t1 = omp_get_wtime();
        if((vector_core_threshold_end - vector_core_threshold_start) > 0)
            vector_core_per_vertex_kernel_all_active(_graph, vector_core_threshold_start,
                                                     vector_core_threshold_end, edge_op, vertex_preprocess_op,
                                                     vertex_postprocess_op, _first_edge, _shard_shift,
                                                     _outgoing_graph_is_stored);
        t2 = omp_get_wtime();
        vc_time += t2 - t1;

        t1 = omp_get_wtime();
        if((collective_threshold_end - collective_threshold_start) > 0)
            ve_collective_vertex_processing_kernel_all_active(_graph, collective_threshold_start, collective_threshold_end,
                                                              collective_edge_op, collective_vertex_preprocess_op,
                                                              collective_vertex_postprocess_op, _first_edge, _shard_shift,
                                                              _outgoing_graph_is_stored);
        t2 = omp_get_wtime();
        collective_time += t2 - t1;
    }
    else
    {
        t1 = omp_get_wtime();
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
        t2 = omp_get_wtime();
        ve_time += t2 - t1;

        t1 = omp_get_wtime();
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
        t2 = omp_get_wtime();
        vc_time += t2 - t1;

        t1 = omp_get_wtime();
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
        t2 = omp_get_wtime();
        collective_time += t2 - t1;
    }

    #pragma omp master // save all stats at once
    {
        wall_time = ve_time + vc_time + collective_time;
        size_t work = 0;
        if(_frontier.type == ALL_ACTIVE_FRONTIER)
            work = _graph.get_edges_count();
        else
            work = _frontier.get_vector_engine_part_neighbours_count() +
                   _frontier.get_vector_core_part_neighbours_count() +
                   _frontier.get_collective_part_neighbours_count();

        performance_stats.fast_update_advance_stats(wall_time, ve_time, vc_time, collective_time,
                                                    work*INT_ELEMENTS_PER_EDGE*sizeof(int), work);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
