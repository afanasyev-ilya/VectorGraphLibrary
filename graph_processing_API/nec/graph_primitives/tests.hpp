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

    DelayedWriteNEC delayed_write;
    delayed_write.init();

    #ifdef __USE_NEC_SX_AURORA__
    ftrace_region_begin("32-length tested region advance");
    #endif

    #pragma omp barrier
    double t1 = omp_get_wtime();
    #pragma omp barrier

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
    ftrace_region_end("32-length tested region advance");
    #endif

    #pragma omp barrier
    double t2 = omp_get_wtime();
    #pragma omp master
    {
        double work = 32 * (last_border - first_border);
        cout << "32-len time: " << (t2 - t1) * 1000.0 << " ms" << endl;
        cout << "32-len TES work: " << work << endl;
        cout << "32-len TEST 3) BW: " << sizeof(int)*6.0*work/((t2-t1)*1e9) << " GB/s" << endl << endl;
    };
    #pragma omp barrier


    // testing with 256 vector length just to demonstrate that high bandwidth is possible

    #ifdef __USE_NEC_SX_AURORA__
    ftrace_region_begin("256-length tested region advance");
    #endif

    #pragma omp barrier
    t1 = omp_get_wtime();
    #pragma omp barrier

    #pragma omp for schedule(static)
    for (int front_pos = first_border; front_pos < last_border; front_pos ++)
    {
        #pragma _NEC collapse
        #pragma _NEC interchange
        for(int vertex_shift = 0; vertex_shift < 1; vertex_shift++)
        {
            int src_id = front_pos + vertex_shift;
            const long long int start = _vertex_pointers[src_id];

            #pragma _NEC vector
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC gather_reorder
            for (int local_edge_pos = 0; local_edge_pos < 256; local_edge_pos++)
            {
                const long long int global_edge_pos = start + local_edge_pos;
                const int vector_index = local_edge_pos + 256 * vertex_shift;
                const int dst_id = _adjacent_ids[global_edge_pos];

                edge_op(src_id, dst_id, local_edge_pos, global_edge_pos, vector_index, delayed_write);
            }
        }
    }

    #ifdef __USE_NEC_SX_AURORA__
    ftrace_region_end("256-length tested region advance");
    #endif

    #pragma omp barrier
    t2 = omp_get_wtime();
    #pragma omp master
    {
        double work = 256 * (last_border - first_border);
        cout << "256-len time: " << (t2 - t1) * 1000.0 << " ms" << endl;
        cout << "256-len TES work: " << work << endl;
        cout << "256-len TEST 3) BW: " << sizeof(int)*6.0*work/((t2-t1)*1e9) << " GB/s" << endl << endl;
    };
    #pragma omp barrier
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
