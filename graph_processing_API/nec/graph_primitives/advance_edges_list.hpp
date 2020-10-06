#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename EdgeOperation>
void GraphPrimitivesNEC::advance_worker(EdgesListGraph &_graph,
                                        EdgeOperation &&edge_op)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
        #pragma omp barrier
        double t1 = omp_get_wtime();
        #pragma omp barrier
    #endif

    LOAD_EDGES_LIST_GRAPH_DATA(_graph);

    #pragma _NEC novector
    #pragma omp for schedule(static, 1)
    for(long long vec_start = 0; vec_start < edges_count; vec_start += VECTOR_LENGTH)
    {
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma _NEC gather_reorder
        for(int i = 0; i < VECTOR_LENGTH; i ++)
        {
            long long global_edge_pos = vec_start + i;
            if(global_edge_pos < edges_count)
            {
                int vector_index = i;
                int src_id = src_ids[global_edge_pos];
                int dst_id = dst_ids[global_edge_pos];
                edge_op(src_id, dst_id, global_edge_pos, vector_index);
            }
        }
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
        #pragma omp barrier
        double t2 = omp_get_wtime();
        #pragma omp master
        {
            double work = edges_count;
            cout << "Advance(el) time: " << (t2 - t1)*1000.0 << " ms" << endl;
            cout << "Advance(el) BW: " << sizeof(int)*INT_ELEMENTS_PER_EDGE*work/((t2-t1)*1e9) << " GB/s" << endl;
        };
        #pragma omp barrier
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

