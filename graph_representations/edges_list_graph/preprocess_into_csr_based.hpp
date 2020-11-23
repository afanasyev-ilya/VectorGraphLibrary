#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void EdgesListGraph::preprocess_into_csr_based(int *_work_buffer, vgl_sort_indexes *_sort_buffer)
{
    bool work_buffer_was_allocated = false;
    if(_work_buffer == NULL)
    {
        work_buffer_was_allocated = true;
        MemoryAPI::allocate_array(&_work_buffer, this->edges_count);
    }

    bool sort_buffer_was_allocated = false;
    if(_sort_buffer == NULL)
    {
        sort_buffer_was_allocated = true;
        MemoryAPI::allocate_array(&_sort_buffer, this->edges_count);
    }

    // init sort indexes
    #pragma _NEC ivdep
    #pragma omp parallel for
    for(int i = 0; i < this->edges_count; i++)
    {
        _sort_buffer[i] = i;
    }

    // sort src_ids
    Timer tm;
    tm.start();
    Sorter::sort(src_ids, _sort_buffer, this->edges_count, SORT_ASCENDING);
    tm.end();
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_stats("EdgesListGraph sorting (to CSR) time");
    #endif

    // reorder dst_ids
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma _NEC gather_reorder
    #pragma omp parallel for
    for(long long edge_pos = 0; edge_pos < this->edges_count; edge_pos++)
    {
        _work_buffer[edge_pos] = dst_ids[_sort_buffer[edge_pos]];
    }

    #pragma _NEC ivdep
    #pragma omp parallel for
    for(long long edge_pos = 0; edge_pos < this->edges_count; edge_pos++)
    {
        dst_ids[edge_pos] = _work_buffer[edge_pos];
    }

    if(work_buffer_was_allocated)
    {
        MemoryAPI::free_array(_work_buffer);
    }
    if(sort_buffer_was_allocated)
    {
        MemoryAPI::free_array(_sort_buffer);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
