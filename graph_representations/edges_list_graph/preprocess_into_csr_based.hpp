#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_ASL__
template <typename _TVertexValue, typename _TEdgeWeight>
void EdgesListGraph<_TVertexValue, _TEdgeWeight>::preprocess_into_csr_based(int *_work_buffer, asl_int_t *_asl_buffer)
{
    bool work_buffer_was_allocated = false;
    if(_work_buffer == NULL)
    {
        work_buffer_was_allocated = true;
        MemoryAPI::allocate_array(&_work_buffer, this->edges_count);
    }

    bool asl_buffer_was_allocated = false;
    if(_asl_buffer == NULL)
    {
        asl_buffer_was_allocated = true;
        MemoryAPI::allocate_array(&_asl_buffer, this->edges_count);
    }

    // init sort indexes
    #pragma _NEC ivdep
    #pragma omp parallel for
    for(int i = 0; i < this->edges_count; i++)
    {
        _asl_buffer[i] = i;
    }

    // initialize ASL sorting library
    asl_sort_t hnd;
    ASL_CALL(asl_library_initialize());
    ASL_CALL(asl_sort_create_i32(&hnd, ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO));

    // sort src_ids
    double t1 = omp_get_wtime();
    ASL_CALL(asl_sort_execute_i32(hnd, this->edges_count, src_ids, _asl_buffer, src_ids, _asl_buffer));
    double t2 = omp_get_wtime();
    cout << "edges list graph sorting time: " << t2 - t1 << " sec" << endl;

    ASL_CALL(asl_sort_destroy(hnd));
    ASL_CALL(asl_library_finalize());

    // reorder dst_ids
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma _NEC gather_reorder
    #pragma omp parallel for
    for(long long edge_pos = 0; edge_pos < this->edges_count; edge_pos++)
    {
        _work_buffer[edge_pos] = dst_ids[_asl_buffer[edge_pos]];
    }

    #pragma _NEC ivdep
    #pragma omp parallel for
    for(long long edge_pos = 0; edge_pos < this->edges_count; edge_pos++)
    {
        dst_ids[edge_pos] = _work_buffer[edge_pos];
    }

    // allocate weights buffer if work_buffer has smaller size
    _TEdgeWeight *weights_buffer;
    bool weights_buffer_was_allocated = false;
    if(sizeof(int) < sizeof(_TEdgeWeight))
    {
        weights_buffer_was_allocated = true;
        MemoryAPI::allocate_array(&weights_buffer, this->edges_count);
    }
    else
    {
        weights_buffer = (_TEdgeWeight *)_work_buffer;
    }

    // reorder weights
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma _NEC gather_reorder
    #pragma omp parallel for
    for(long long edge_pos = 0; edge_pos < this->edges_count; edge_pos++)
    {
        weights_buffer[edge_pos] = weights[_asl_buffer[edge_pos]];
    }

    #pragma _NEC ivdep
    #pragma omp parallel for
    for(long long edge_pos = 0; edge_pos < this->edges_count; edge_pos++)
    {
        weights[edge_pos] = weights_buffer[edge_pos];
    }

    // free all buffers if needed
    if(weights_buffer_was_allocated)
    {
        MemoryAPI::free_array(weights_buffer);
    }
    if(work_buffer_was_allocated)
    {
        MemoryAPI::free_array(_work_buffer);
    }
    if(asl_buffer_was_allocated)
    {
        MemoryAPI::free_array(_asl_buffer);
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
