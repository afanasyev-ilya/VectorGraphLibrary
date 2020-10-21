#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphAbstractionsNEC::vector_engine_per_vertex_kernel_sparse(UndirectedCSRGraph &_graph,
                                                                  FrontierNEC &_frontier,
                                                                  EdgeOperation edge_op,
                                                                  VertexPreprocessOperation vertex_preprocess_op,
                                                                  VertexPostprocessOperation vertex_postprocess_op,
                                                                  const int _first_edge)
{
    Timer tm;
    tm.start();

    TraversalDirection traversal = current_traversal_direction;
    int storage = CSR_STORAGE;

    DelayedWriteNEC delayed_write;
    delayed_write.init();

    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);
    int *frontier_ids = &(_frontier.get_ids()[0]);
    int frontier_segment_size = _frontier.get_vector_engine_part_size();

    for (int front_pos = 0; front_pos < frontier_segment_size; front_pos++)
    {
        const int src_id = frontier_ids[front_pos];

        const long long int start = vertex_pointers[src_id];
        const long long int end = vertex_pointers[src_id + 1];
        const int connections_count = end - start;

        vertex_preprocess_op(src_id, connections_count, 0, delayed_write);

        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma _NEC gather_reorder
        #pragma omp for schedule(static)
        for (int local_edge_pos = 0; local_edge_pos < connections_count; local_edge_pos++)
        {
            const long long int internal_edge_pos = start + local_edge_pos;
            const int vector_index = get_vector_index(local_edge_pos);
            const int dst_id = adjacent_ids[internal_edge_pos];
            const long long external_edge_pos = traversal * direction_shift + storage * edges_count + internal_edge_pos;

            edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index, delayed_write);
        }

        vertex_postprocess_op(src_id, connections_count, 0, delayed_write);
    }

    tm.end();
    performance_stats.update_advance_ve_part_time(tm);

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    long long work = _frontier.get_vector_engine_part_neighbours_count();
    tm.print_bandwidth_stats("Advance (ve)", work, INT_ELEMENTS_PER_EDGE*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphAbstractionsNEC::vector_core_per_vertex_kernel_sparse(UndirectedCSRGraph &_graph,
                                                                FrontierNEC &_frontier,
                                                                EdgeOperation edge_op,
                                                                VertexPreprocessOperation vertex_preprocess_op,
                                                                VertexPostprocessOperation vertex_postprocess_op,
                                                                const int _first_edge)
{
    Timer tm;
    tm.start();

    TraversalDirection traversal = current_traversal_direction;
    int storage = CSR_STORAGE;

    DelayedWriteNEC delayed_write;
    delayed_write.init();

    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);
    int *frontier_ids = &(_frontier.get_ids()[_frontier.get_vector_engine_part_size()]);
    int frontier_segment_size = _frontier.get_vector_core_part_size();

    #pragma omp for schedule(static, 8)
    for (int front_pos = 0; front_pos < frontier_segment_size; front_pos++)
    {
        const int src_id = frontier_ids[front_pos];

        const long long int start = vertex_pointers[src_id];
        const long long int end = vertex_pointers[src_id + 1];
        const int connections_count = end - start;

        vertex_preprocess_op(src_id, connections_count, 0, delayed_write);

        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma _NEC gather_reorder
        for (int local_edge_pos = 0; local_edge_pos < connections_count; local_edge_pos++)
        {
            const long long int internal_edge_pos = start + local_edge_pos;
            const int vector_index = get_vector_index(local_edge_pos);
            const int dst_id = adjacent_ids[internal_edge_pos];
            const long long external_edge_pos = traversal * direction_shift + storage * edges_count + internal_edge_pos;

            edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index, delayed_write);
        }

        vertex_postprocess_op(src_id, connections_count, 0, delayed_write);
    }

    tm.end();
    performance_stats.update_advance_vc_part_time(tm);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    long long work = _frontier.get_vector_core_part_neighbours_count();
    tm.print_bandwidth_stats("Advance (vc)", work, INT_ELEMENTS_PER_EDGE*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <algorithm>

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphAbstractionsNEC::collective_vertex_processing_kernel_sparse(UndirectedCSRGraph &_graph,
                                                                      FrontierNEC &_frontier,
                                                                      const int _first_vertex,
                                                                      const int _last_vertex,
                                                                      EdgeOperation edge_op,
                                                                      VertexPreprocessOperation vertex_preprocess_op,
                                                                      VertexPostprocessOperation vertex_postprocess_op,
                                                                      const int _first_edge)
{
    Timer tm;
    tm.start();

    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);
    int *frontier_ids = &(_frontier.get_ids()[_frontier.get_vector_core_part_size() + _frontier.get_vector_engine_part_size()]);
    int frontier_segment_size = _frontier.get_collective_part_size();

    TraversalDirection traversal = current_traversal_direction;
    int storage = CSR_STORAGE;

    long long int reg_start[VECTOR_LENGTH];
    long long int reg_end[VECTOR_LENGTH];
    int reg_connections[VECTOR_LENGTH];

    #pragma _NEC vreg(reg_start)
    #pragma _NEC vreg(reg_end)
    #pragma _NEC vreg(reg_connections)

    #pragma _NEC vector
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        reg_start[i] = 0;
        reg_end[i] = 0;
        reg_connections[i] = 0;
    }

    DelayedWriteNEC delayed_write;
    delayed_write.init();

    #pragma omp for schedule(static, 8)
    for(int front_pos = 0; front_pos < frontier_segment_size; front_pos += VECTOR_LENGTH)
    {
        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            if((front_pos + i) < frontier_segment_size)
            {
                int src_id = frontier_ids[front_pos + i];
                reg_start[i] = vertex_pointers[src_id];
                reg_end[i] = vertex_pointers[src_id + 1];
                reg_connections[i] = reg_end[i] - reg_start[i];
                vertex_preprocess_op(src_id, reg_connections[i], i, delayed_write);
            }
            else
            {
                reg_start[i] = 0;
                reg_end[i] = 0;
                reg_connections[i] = 0;
            }
        }

        int max_connections = 0;
        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            if(max_connections < reg_connections[i])
            {
                max_connections = reg_connections[i];
            }
        }

        if(max_connections > 0)
        {
            for (int edge_pos = _first_edge; edge_pos < max_connections; edge_pos++)
            {
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                for (int i = 0; i < VECTOR_LENGTH; i++)
                {
                    if (((front_pos + i) < frontier_segment_size) && (edge_pos < reg_connections[i]))
                    {
                        const int src_id = frontier_ids[front_pos + i];
                        const int vector_index = i;
                        const long long int internal_edge_pos = reg_start[i] + edge_pos;
                        const int local_edge_pos = edge_pos;
                        const int dst_id = adjacent_ids[internal_edge_pos];
                        const long long external_edge_pos = traversal * direction_shift + storage * edges_count + internal_edge_pos;

                        edge_op(src_id, dst_id, local_edge_pos, external_edge_pos, vector_index, delayed_write);
                    }
                }
            }

            #pragma _NEC vector
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                if ((front_pos + i) < frontier_segment_size)
                {
                    int src_id = frontier_ids[front_pos + i];
                    vertex_postprocess_op(src_id, reg_connections[i], i, delayed_write);
                }
            }
        }
    }

    tm.end();
    performance_stats.update_advance_collective_part_time(tm);
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    long long work = _frontier.get_collective_part_neighbours_count();
    tm.print_bandwidth_stats("Advance (collective)", work, INT_ELEMENTS_PER_EDGE*sizeof(int));
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
