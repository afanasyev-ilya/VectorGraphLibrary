#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation>
void GraphPrimitivesMulticore::advance(UndirectedGraph &_graph,
                                       FrontierMulticore &_frontier,
                                       EdgeOperation &&edge_op)
{
    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);

    const long long int *_vertex_pointers = vertex_pointers;
    const int *_adjacent_ids = adjacent_ids;
    const int *_ve_adjacent_ids = ve_adjacent_ids;
    int *_frontier_flags = _frontier.flags;

    const int vector_engine_threshold_start = 0;
    const int vector_engine_threshold_end = _graph.get_vector_engine_threshold_vertex();
    const int vector_core_threshold_start = _graph.get_vector_engine_threshold_vertex();
    const int vector_core_threshold_end = _graph.get_vector_core_threshold_vertex();
    const int collective_threshold_start = _graph.get_vector_core_threshold_vertex();
    const int collective_threshold_end = _graph.get_vertices_count();

    #pragma omp parallel
    {
        #pragma ivdep
        #pragma vector
        #pragma omp for schedule(dynamic)
        for(int front_pos = vector_engine_threshold_start; front_pos < vector_engine_threshold_end; front_pos++)
        {
            const int src_id = front_pos;

            const long long int start = _vertex_pointers[src_id];
            const long long int end = _vertex_pointers[src_id + 1];
            const int connections_count = end - start;

            //vertex_preprocess_op(src_id, connections_count, 0, delayed_write);

            for (int local_edge_pos = 0; local_edge_pos < connections_count; local_edge_pos++)
            {
                const long long int global_edge_pos = start + local_edge_pos;
                const int vector_index = get_vector_index(local_edge_pos);
                const int dst_id = _adjacent_ids[global_edge_pos];

                edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index);
            }

            //vertex_postprocess_op(src_id, connections_count, 0, delayed_write);
        }

        #pragma ivdep
        #pragma vector
        #pragma omp for schedule(guided)
        for(int front_pos = vector_core_threshold_start; front_pos < vector_core_threshold_end; front_pos++)
        {
            const int src_id = front_pos;

            const long long int start = _vertex_pointers[src_id];
            const long long int end = _vertex_pointers[src_id + 1];
            const int connections_count = end - start;

            //vertex_preprocess_op(src_id, connections_count, 0, delayed_write);

            for (int local_edge_pos = 0; local_edge_pos < connections_count; local_edge_pos++)
            {
                const long long int global_edge_pos = start + local_edge_pos;
                const int vector_index = get_vector_index(local_edge_pos);
                const int dst_id = _adjacent_ids[global_edge_pos];

                edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index);
            }

            //vertex_postprocess_op(src_id, connections_count, 0, delayed_write);
        }

        #pragma omp for schedule(static, 1024)
        for(int front_pos = collective_threshold_start; front_pos < collective_threshold_end; front_pos++)
        {
            const int src_id = front_pos;

            const long long int start = _vertex_pointers[src_id];
            const long long int end = _vertex_pointers[src_id + 1];
            const int connections_count = end - start;

            //vertex_preprocess_op(src_id, connections_count, 0, delayed_write);

            for (int local_edge_pos = 0; local_edge_pos < connections_count; local_edge_pos++)
            {
                const long long int global_edge_pos = start + local_edge_pos;
                const int vector_index = get_vector_index(local_edge_pos);
                const int dst_id = _adjacent_ids[global_edge_pos];

                edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index);
            }

            //vertex_postprocess_op(src_id, connections_count, 0, delayed_write);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
