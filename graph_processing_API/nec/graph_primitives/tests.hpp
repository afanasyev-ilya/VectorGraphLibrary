/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
#include <ftrace.h>
#endif

template <typename EdgeOperation, typename VertexPreprocessOperation,
        typename VertexPostprocessOperation>
void GraphPrimitivesNEC::my_test(const long long *_vertex_pointers, const int *_adjacent_ids,
                                 const int _first_vertex,
                                 const int _last_vertex,
                                 EdgeOperation edge_op,
                                 VertexPreprocessOperation vertex_preprocess_op,
                                 VertexPostprocessOperation vertex_postprocess_op,
                                 const int _first_edge)
{
    int first_border = 0;
    int last_border = _last_vertex;
    /*for (int front_pos = _first_vertex; front_pos < _last_vertex - 1; front_pos++)
    {
        const int src_id = front_pos;

        const long long int start = _vertex_pointers[src_id];
        const long long int end = _vertex_pointers[src_id + 1];
        const int connections_count = end - start;

        const long long int next_start = _vertex_pointers[src_id + 1];
        const long long int next_end = _vertex_pointers[src_id + 2];
        const int next_connections_count = next_end - next_start;

        if((next_connections_count < VECTOR_LENGTH) && (connections_count >= VECTOR_LENGTH))
        {
            top_border = src_id;
        }

        if((next_connections_count < 128) && (connections_count >= 128))
        {
            down_border = src_id;
        }
    }*/

    #pragma omp master
    {
        cout << "borders: (" << first_border << " - " << last_border << ")" << endl;
    }

    #ifdef __USE_NEC_SX_AURORA__
    //ftrace_region_begin("test reg");
    #endif

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    #pragma omp barrier
    double t1 = omp_get_wtime();
    #pragma omp barrier
    #endif

    int reg_res[VECTOR_LENGTH];
    #pragma _NEC vreg(reg_res)

    DelayedWriteNEC delayed_write;
    delayed_write.init();

    #pragma omp for schedule(static)
    for (int front_pos = first_border; front_pos < last_border; front_pos += 8)
    {
        #pragma _NEC collapse
        #pragma _NEC interchange
        for(int vertex_shift = 0; vertex_shift < 8; vertex_shift++)
        {
            int src_id = front_pos + vertex_shift;
            const long long int start = _vertex_pointers[src_id];

            #pragma _NEC vector
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC gather_reorder
            for (int local_edge_pos = 0; local_edge_pos < 32; local_edge_pos++)
            {
                const long long int global_edge_pos = start + local_edge_pos;
                const int vector_index = local_edge_pos + 32 * vertex_shift;
                const int dst_id = _adjacent_ids[global_edge_pos];

                edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index, delayed_write);
            }
        }
    }

    #ifdef __USE_NEC_SX_AURORA__
    //ftrace_region_end("test reg");
    #endif

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    #pragma omp barrier
        double t2 = omp_get_wtime();
        #pragma omp master
        {
            INNER_WALL_NEC_TIME += t2 - t1;

            //double work = _vertex_pointers[last_border] - _vertex_pointers[first_border];
            double work = 32 * (last_border - first_border);
            cout << "TEST BANDWIDTH: " << endl;
            cout << "TES work: " << work << endl;
            cout << "TEST 3) time: " << (t2 - t1)*1000.0 << " ms" << endl;
            cout << "TEST 3) BW: " << sizeof(int)*6.0*work/((t2-t1)*1e9) << " GB/s" << endl << endl;
        };
        #pragma omp barrier
    #endif
    int sum = 0;
    for(int i = 0; i < 256; i++)
        sum += reg_res[i];

    cout << "sum " << sum << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
