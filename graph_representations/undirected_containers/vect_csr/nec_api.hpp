#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__USE_NEC_SX_AURORA__) || defined(__USE_MULTICORE__)
void VectorCSRGraph::estimate_nec_thresholds()
{
    Timer tm;
    tm.start();

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
        const int current_size = vertex_pointers[current_id + 1] - vertex_pointers[current_id];

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
            next_size = vertex_pointers[next_id + 1] - vertex_pointers[next_id];
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

    tm.end();
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_stats("Estimate thresholds");
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
