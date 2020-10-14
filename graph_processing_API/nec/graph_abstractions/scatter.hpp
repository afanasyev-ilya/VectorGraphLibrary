#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation>
void GraphAbstractionsNEC::scatter(VectCSRGraph &_graph,
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
    UndirectedCSRGraph *current_direction_graph;

    if(current_traversal_direction != SCATTER)
    {
        throw "Error in GraphAbstractionsNEC::scatter : wrong traversal direction";
    }
    if(_frontier.get_direction() != current_traversal_direction)
    {
        throw "Error in GraphAbstractionsNEC::scatter : wrong frontier direction";
    }
    current_direction_graph = _graph.get_outgoing_graph_ptr();

    if(omp_in_parallel())
    {
        #pragma omp barrier
        advance_worker(*current_direction_graph, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op,
                       collective_edge_op, collective_vertex_preprocess_op, collective_vertex_postprocess_op, 0);
        #pragma omp barrier
    }
    else
    {
        #pragma omp parallel
        {
            advance_worker(*current_direction_graph, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op,
                           collective_edge_op, collective_vertex_preprocess_op, collective_vertex_postprocess_op, 0);
        }
    }
    tm.end();
    performance_stats.update_scatter_time(tm);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation, typename CollectiveEdgeOperation, typename CollectiveVertexPreprocessOperation,
        typename CollectiveVertexPostprocessOperation>
void GraphAbstractionsNEC::scatter(ShardedCSRGraph &_graph,
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

    if(current_traversal_direction != SCATTER)
    {
        throw "Error in GraphAbstractionsNEC::scatter : wrong traversal direction";
    }
    if(_frontier.get_direction() != current_traversal_direction)
    {
        throw "Error in GraphAbstractionsNEC::scatter : wrong frontier direction";
    }

    for(int shard_id = 0; shard_id < _graph.get_shards_number(); shard_id++)
    {
        if(_frontier.get_type() != ALL_ACTIVE_FRONTIER)
        {
            throw "Error in GraphAbstractionsNEC::scatter : sparse/dense frontiers are currently not supported";
        }

        UndirectedCSRGraph *current_shard = _graph.get_outgoing_shard_ptr(shard_id);
        if(omp_in_parallel())
        {
            #pragma omp barrier
            advance_worker(*current_shard, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op,
                           collective_edge_op, collective_vertex_preprocess_op, collective_vertex_postprocess_op, 0);
            #pragma omp barrier
        }
        else
        {
            #pragma omp parallel
            {
                advance_worker(*current_shard, _frontier, edge_op, vertex_preprocess_op, vertex_postprocess_op,
                               collective_edge_op, collective_vertex_preprocess_op, collective_vertex_postprocess_op, 0);
            }
        }
    }

    tm.end();
    performance_stats.update_scatter_time(tm);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
