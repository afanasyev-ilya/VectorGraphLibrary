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
    ftrace_region_begin("test reg");
    #endif

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    #pragma omp barrier
    double t1 = omp_get_wtime();
    #pragma omp barrier
    #endif

    int reg_res[VECTOR_LENGTH];

    DelayedWriteNEC delayed_write;
    delayed_write.init();

    #pragma omp for schedule(static)
    for (int front_pos = first_border; front_pos < last_border - 256; front_pos += 4)
    {
        #pragma _NEC vector
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        for(int i = 0; i < 256; i++)
        {
            const int virtual_warp_id = i >> 6;
            const int position_in_virtual_warp = i & (4 - 1);

            int src_id = front_pos + 0;// virtual_warp_id;

            const long long int start = _vertex_pointers[src_id];
            //const long long int end = _vertex_pointers[src_id + 1];
            //const int connections_count = end - start;

            const long long int global_edge_pos = start + i;
            //const int vector_index = i;
            const int dst_id = _adjacent_ids[global_edge_pos];

            reg_res[i] += dst_id;

            //edge_op(src_id, dst_id, position_in_virtual_warp, global_edge_pos, vector_index, delayed_write);
        }
    }

    #ifdef __USE_NEC_SX_AURORA__
    ftrace_region_end("test reg");
    #endif

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    #pragma omp barrier
        double t2 = omp_get_wtime();
        #pragma omp master
        {
            INNER_WALL_NEC_TIME += t2 - t1;

            double work = _vertex_pointers[last_border] - _vertex_pointers[first_border];
            cout << "TEST BANDWIDTH: " << endl;
            cout << "TES work: " << work << endl;
            cout << "TEST 3) time: " << (t2 - t1)*1000.0 << " ms" << endl;
            cout << "TEST 3) BW: " << sizeof(int)*2.0*work/((t2-t1)*1e9) << " GB/s" << endl << endl;
        };
        #pragma omp barrier
    #endif
    int sum = 0;
    for(int i = 0; i < 256; i++)
        sum += reg_res[i];

    cout << "sum " << sum << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
