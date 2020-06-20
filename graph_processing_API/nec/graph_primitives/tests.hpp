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
    int top_border = 0;
    int down_border = 0;
    for (int front_pos = _first_vertex; front_pos < _last_vertex - 1; front_pos++)
    {
        const int src_id = front_pos;

        const long long int start = _vertex_pointers[src_id];
        const long long int end = _vertex_pointers[src_id + 1];
        const int connections_count = end - start;

        const long long int next_start = _vertex_pointers[src_id + 1];
        const long long int next_end = _vertex_pointers[src_id + 2];
        const int next_connections_count = next_end - next_start;

        if((next_connections_count < 64) && (connections_count >= 64))
        {
            top_border = src_id;
        }

        if((next_connections_count < 32) && (connections_count >= 32))
        {
            down_border = src_id;
        }
    }

    #pragma omp master
    {
        cout << "borders: (" << top_border << " - " << down_border << ")" << endl;
    }


    #ifdef __USE_NEC_SX_AURORA__
    ftrace_region_begin("test reg");
    #endif

    float reg_res[VECTOR_LENGTH];
    for(int i = 0; i < 256; i++)
        reg_res[i] = 0.0;
    #pragma _NEC vreg(reg_res)

    DelayedWriteNEC delayed_write;
    delayed_write.init();

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    #pragma omp barrier
    double t1 = omp_get_wtime();
    #pragma omp barrier
    #endif

    #pragma omp for schedule(static)
    for (int front_pos = top_border; front_pos < down_border; front_pos += 4)
    {
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        for(int ij = 0; ij < 8*32; ij++)
        {
            int vertex_shift = ij / 256;
            int local_edge_pos = ij % 256;

            int src_id = front_pos + vertex_shift;
            const long long int start = _vertex_pointers[src_id];

            const long long int global_edge_pos = start + local_edge_pos;
            const int vector_index = ij;
            const int dst_id = _adjacent_ids[global_edge_pos];

            reg_res[ij] += dst_id;
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

            double work = 32*(down_border-top_border);//_vertex_pointers[down_border] - _vertex_pointers[top_border];
            cout << "TEST BANDWIDTH: " << endl;
            cout << "TES work: " << work << endl;
            cout << "TEST 3) time: " << (t2 - t1)*1000.0 << " ms" << endl;
            cout << "NEW kazu verc TEST 3) BW: " << sizeof(int)*2.0*work/((t2-t1)*1e9) << " GB/s" << endl << endl;
        };
        #pragma omp barrier
    #endif

    int res = 0;
    for(int i = 0; i < 256; i++)
    {
        res += reg_res[i];
    }
    #pragma omp single
    {
        cout << "res: " << res << endl;
    };
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
