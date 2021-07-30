#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation>
void GraphAbstractionsNEC::gather(VectCSRGraph &_graph,
                                  FrontierNEC &_frontier,
                                  EdgeOperation &&edge_op,
                                  VertexPreprocessOperation &&vertex_preprocess_op,
                                  VertexPostprocessOperation &&vertex_postprocess_op,
                                  CollectiveEdgeOperation &&collective_edge_op,
                                  CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                                  CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op)
{
    Timer tm;
    tm.start();
    VectorCSRGraph *current_direction_graph;

    if(current_traversal_direction != GATHER)
    {
        throw "Error in GraphAbstractionsNEC::gather : wrong traversal direction";
    }
    if(_frontier.get_direction() != current_traversal_direction)
    {
        throw "Error in GraphAbstractionsNEC::gather : wrong frontier direction";
    }
    current_direction_graph = _graph.get_incoming_graph_ptr();

    bool outgoing_graph_is_stored = _graph.outgoing_is_stored();
    bool inner_mpi_processing = false;
    #ifdef __USE_MPI__
    inner_mpi_processing = true;
    #endif
    if(omp_in_parallel())
    {
        #pragma omp barrier
        advance_worker(*current_direction_graph, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op,
                       collective_edge_op, collective_vertex_preprocess_op, collective_vertex_postprocess_op, 0, 0,
                       outgoing_graph_is_stored, inner_mpi_processing);
        #pragma omp barrier
    }
    else
    {
        #pragma omp parallel
        {
            advance_worker(*current_direction_graph, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op,
                           collective_edge_op, collective_vertex_preprocess_op, collective_vertex_postprocess_op, 0, 0,
                           outgoing_graph_is_stored, inner_mpi_processing);
        }
    }

    tm.end();
    performance_stats.update_gather_time(tm);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation>
void GraphAbstractionsNEC::gather(VectCSRGraph &_graph,
                                  FrontierNEC &_frontier,
                                  EdgeOperation &&edge_op)
{
    gather(_graph, _frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
           edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation>
void GraphAbstractionsNEC::gather(ShardedCSRGraph &_graph,
                                  FrontierNEC &_frontier,
                                  EdgeOperation &&edge_op,
                                  VertexPreprocessOperation &&vertex_preprocess_op,
                                  VertexPostprocessOperation &&vertex_postprocess_op,
                                  CollectiveEdgeOperation &&collective_edge_op,
                                  CollectiveVertexPreprocessOperation &&collective_vertex_preprocess_op,
                                  CollectiveVertexPostprocessOperation &&collective_vertex_postprocess_op)
{
    if(omp_in_parallel())
    {
        Timer tm;
        tm.start();

        if(current_traversal_direction != GATHER)
        {
            throw "Error in GraphAbstractionsNEC::gather : wrong traversal direction";
        }
        if(_frontier.get_direction() != current_traversal_direction)
        {
            throw "Error in GraphAbstractionsNEC::gather : wrong frontier direction";
        }
        if(_frontier.get_type() != ALL_ACTIVE_FRONTIER)
        {
            throw "Error in GraphAbstractionsNEC::gather : sparse/dense frontiers are currently not supported";
        }

        bool outgoing_graph_is_stored = true;

        int first_shard = 0;
        int shards_step = 1;
        #ifdef __USE_MPI__
        first_shard = vgl_library_data.get_mpi_rank();
        shards_step = vgl_library_data.get_mpi_proc_num();
        #endif

        for(int shard_id =  first_shard; shard_id < _graph.get_shards_number(); shard_id += shards_step)
        {
            VectorCSRGraph *current_shard = _graph.get_incoming_shard_ptr(shard_id);

            // prepare user data for current shard
            for(auto& current_container : user_data_containers)
            {
                current_container->reorder_from_original_to_shard(current_traversal_direction, shard_id);
            }

            long long shard_shift = _graph.get_shard_shift(shard_id, current_traversal_direction);
            bool inner_mpi_processing = false;
            advance_worker(*current_shard, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op,
                           collective_edge_op, collective_vertex_preprocess_op, collective_vertex_postprocess_op, 0, shard_shift,
                           outgoing_graph_is_stored, inner_mpi_processing);

            // reorder user data back
            for(auto& current_container : user_data_containers)
            {
                current_container->reorder_from_shard_to_original(current_traversal_direction, shard_id);
            }
        }

        tm.end();
        performance_stats.update_scatter_time(tm);
    }
    else
    {
        #pragma omp parallel
        {
            gather(_graph, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op, collective_edge_op,
                    collective_vertex_preprocess_op, collective_vertex_postprocess_op);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation>
void GraphAbstractionsNEC::gather(ShardedCSRGraph &_graph,
                                   FrontierNEC &_frontier,
                                   EdgeOperation &&edge_op)
{
    gather(_graph, _frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP,
            edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
