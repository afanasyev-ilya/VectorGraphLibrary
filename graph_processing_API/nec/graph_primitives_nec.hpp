/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename InitOperation>
void GraphPrimitivesNEC::init(int _size, InitOperation init_op)
{
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp for schedule(static)
    for(int src_id = 0; src_id < _size; src_id++)
    {
        init_op(src_id);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation>
void GraphPrimitivesNEC::advance(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                 FrontierNEC &_frontier,
                                 int large_threshold_vertex,
                                 int medium_threshold_vertex,
                                 EdgeOperation edge_op)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    const long long int *vertex_pointers = outgoing_ptrs;
    const int *adjacent_ids = outgoing_ids;
    const _TEdgeWeight *adjacent_weights = outgoing_weights;

    for (int front_pos = 0; front_pos < large_threshold_vertex; front_pos++)
    {
        const int src_id = front_pos;
        const long long int start = vertex_pointers[src_id];
        const long long int end = vertex_pointers[src_id + 1];
        const int connections_count = end - start;

        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma omp for schedule(static)
        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            const long long int global_edge_pos = start + edge_pos;
            const int local_edge_pos = edge_pos;
            const int vector_index = edge_pos % VECTOR_LENGTH;
            int dst_id = adjacent_ids[global_edge_pos];

            edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index);
        }
    }

    #pragma omp for schedule(static, 8)
    for (int front_pos = large_threshold_vertex; front_pos < medium_threshold_vertex; front_pos ++)
    {
        const int src_id = front_pos;
        const long long int start = vertex_pointers[src_id];
        const long long int end = vertex_pointers[src_id + 1];
        const int connections_count = end - start;

        for(int edge_vec_pos = 0; edge_vec_pos < connections_count - VECTOR_LENGTH; edge_vec_pos += VECTOR_LENGTH)
        {
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                const long long int global_edge_pos = start + edge_vec_pos + i;
                const int local_edge_pos = edge_vec_pos + i;
                const int vector_index = i;
                const int dst_id = adjacent_ids[global_edge_pos];

                edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index);
            }
        }

        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        for(int i = connections_count - VECTOR_LENGTH; i < connections_count; i++)
        {
            const long long int global_edge_pos = start + i;
            const int local_edge_pos = i;
            const int vector_index = i - (connections_count - VECTOR_LENGTH);
            const int dst_id = adjacent_ids[global_edge_pos];

            edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index);
        }
    }

    long long int reg_start[VECTOR_LENGTH];
    long long int reg_end[VECTOR_LENGTH];
    int reg_connections[VECTOR_LENGTH];

    #pragma _NEC vreg(reg_start)
    #pragma _NEC vreg(reg_end)
    #pragma _NEC vreg(reg_connections)

    #pragma omp for schedule(static, 1)
    for(int front_pos = medium_threshold_vertex; front_pos < vertices_count; front_pos += VECTOR_LENGTH)
    {
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            int src_id = front_pos + i;
            if(src_id < vertices_count)
            {
                reg_start[i] = vertex_pointers[src_id];
                reg_end[i] = vertex_pointers[src_id + 1];
                reg_connections[i] = reg_end[i] - reg_start[i];
            }
            else
            {
                reg_start[i] = 0;
                reg_end[i] = 0;
                reg_connections[i] = 0;
            }
        }

        int max_connections = vertex_pointers[front_pos + 1] - vertex_pointers[front_pos];

        for(int edge_pos = 0; edge_pos < max_connections; edge_pos++)
        {
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                const int src_id = front_pos + i;
                if((src_id < vertices_count) && (edge_pos < reg_connections[i]))
                {
                    const int vector_index = i;
                    const long long int global_edge_pos = reg_start[i] + edge_pos;
                    const int local_edge_pos = edge_pos;
                    const int dst_id = adjacent_ids[global_edge_pos];

                    edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index);
                }
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
