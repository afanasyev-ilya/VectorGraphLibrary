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
                                 EdgeOperation edge_op)
{
    #pragma omp barrier

    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    const long long int *vertex_pointers = outgoing_ptrs;
    const int *adjacent_ids = outgoing_ids;
    const _TEdgeWeight *adjacent_weights = outgoing_weights;

    int large_threshold_start, large_threshold_end;
    int medium_threshold_start, medium_threshold_end;
    int small_threshold_start, small_threshold_end;
    _frontier.split_sorted_frontier(vertex_pointers, large_threshold_start, large_threshold_end,
                                    medium_threshold_start, medium_threshold_end,
                                    small_threshold_start, small_threshold_end);

    /*#pragma omp single
    {
        cout << large_threshold_start << " " <<  large_threshold_end << endl;
        cout << medium_threshold_start << " " <<  medium_threshold_end << endl;
        cout << small_threshold_start << " " <<  small_threshold_end << endl;
    }*/

    #pragma omp barrier

    #ifdef __PRINT_DETAILED_STATS__
    double t1, t2;
    t1 = omp_get_wtime();
    #endif

    int *frontier_ids = _frontier.frontier_ids;
    int *is_active = _frontier.frontier_flags;
    int *frontier_flags = _frontier.frontier_flags;

    for (int front_pos = large_threshold_start; front_pos < large_threshold_end; front_pos++)
    {
        const int src_id = front_pos; //frontier_ids[front_pos];
        //if(_was_changes[src_id] > 0)
        //{
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
        //}
    }

    #ifdef __PRINT_DETAILED_STATS__
    t2 = omp_get_wtime();
    #pragma omp master
    {
        cout << "first time: " << (t2 - t1)*1000 << " ms" << endl;
        cout << "first BW: " << (sizeof(int)*5.0)*(vertex_pointers[large_threshold_end] - vertex_pointers[large_threshold_start])/((t2-t1)*1e9) << " GB/s" << endl;
    };
    #pragma omp barrier

    t1 = omp_get_wtime();
    #endif

    #pragma omp for schedule(static, 8)
    for (int front_pos = medium_threshold_start; front_pos < medium_threshold_end; front_pos ++)
    {
        const int src_id = front_pos;//frontier_ids[front_pos];
        //if(_was_changes[src_id] > 0)
        //{
            const long long int start = vertex_pointers[src_id];
            const long long int end = vertex_pointers[src_id + 1];
            const int connections_count = end - start;

            for (int edge_vec_pos = 0; edge_vec_pos < connections_count - VECTOR_LENGTH; edge_vec_pos += VECTOR_LENGTH)
            {
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                for (int i = 0; i < VECTOR_LENGTH; i++)
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
            for (int i = connections_count - VECTOR_LENGTH; i < connections_count; i++)
            {
                const long long int global_edge_pos = start + i;
                const int local_edge_pos = i;
                const int vector_index = i - (connections_count - VECTOR_LENGTH);
                const int dst_id = adjacent_ids[global_edge_pos];

                edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index);
            }
        //}
    }

    #ifdef __PRINT_DETAILED_STATS__
    t2 = omp_get_wtime();
    #pragma omp master
    {
        cout << "second time: " << (t2 - t1)*1000 << " ms" << endl;
        cout << "second BW: " << (sizeof(int)*5.0)*(vertex_pointers[medium_threshold_end] - vertex_pointers[medium_threshold_start])/((t2-t1)*1e9) << " GB/s" << endl;
    };
    #pragma omp barrier
    #endif

    #ifdef __PRINT_DETAILED_STATS__
    t1 = omp_get_wtime();
    #endif

    long long int reg_start[VECTOR_LENGTH];
    long long int reg_end[VECTOR_LENGTH];
    int reg_connections[VECTOR_LENGTH];

    #pragma _NEC vreg(reg_start)
    #pragma _NEC vreg(reg_end)
    #pragma _NEC vreg(reg_connections)

    #pragma omp for schedule(static, 1)
    for(int front_pos = small_threshold_start; front_pos < small_threshold_end; front_pos += VECTOR_LENGTH)
    {
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            if((front_pos + i) < small_threshold_end)
            {
                int src_id = front_pos + i;//frontier_ids[front_pos + i];
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
                if(((front_pos + i) < small_threshold_end) && (edge_pos < reg_connections[i]))
                {
                    const int src_id = front_pos + i;//frontier_ids[front_pos + i];
                    //if(_was_changes[src_id] > 0)
                    //{
                        const int vector_index = i;
                        const long long int global_edge_pos = reg_start[i] + edge_pos;
                        const int local_edge_pos = edge_pos;
                        const int dst_id = adjacent_ids[global_edge_pos];

                        edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index);
                    //}
                }
            }
        }
    }

    #ifdef __PRINT_DETAILED_STATS__
    t2 = omp_get_wtime();
    #pragma omp master
    {
        cout << "third time: " << (t2 - t1)*1000 << " ms" << endl;
        cout << "third BW: " << (sizeof(int)*5.0)*(vertex_pointers[small_threshold_end] - vertex_pointers[small_threshold_start])/((t2-t1)*1e9) << " GB/s" << endl << endl;
    };
    #pragma omp barrier
    #endif

    #pragma omp barrier
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
