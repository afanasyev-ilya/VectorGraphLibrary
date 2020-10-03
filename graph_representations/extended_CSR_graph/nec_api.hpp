#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_NEC_SX_AURORA__
template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::estimate_nec_thresholds()
{
    double t1 = omp_get_wtime();

    vector_engine_threshold_vertex = 0;
    vector_core_threshold_vertex = 0;

    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp parallel for schedule(static)
    for(int idx = 0; idx < this->vertices_count; idx++)
    {
        const int current_id = idx;
        const int current_size = outgoing_ptrs[current_id + 1] - outgoing_ptrs[current_id];

        int next_id = 0;
        int next_size = 0;
        if(idx == (this->vertices_count - 1))
        {
            next_id = -1;
            next_size = 0;
        }
        else
        {
            next_id = idx + 1;
            next_size = outgoing_ptrs[next_id + 1] - outgoing_ptrs[next_id];
        }

        if((current_size >= VECTOR_ENGINE_THRESHOLD_VALUE) && (next_size < VECTOR_ENGINE_THRESHOLD_VALUE))
        {
            vector_engine_threshold_vertex = idx + 1;
        }
        else if((current_size >= VECTOR_CORE_THRESHOLD_VALUE) && (next_size < VECTOR_CORE_THRESHOLD_VALUE))
        {
            vector_core_threshold_vertex = idx + 1;
        }
    }

    double t2 = omp_get_wtime();
    cout << "estimate thresholds time: " << t2 - t1 << " sec" << endl;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
