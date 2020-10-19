#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_ASL__
#include <asl.h>
#endif

#ifdef __USE_ASL__
#define vgl_sort_indexes asl_int_t
#endif
#ifndef __USE_ASL__
#define vgl_sort_indexes long long
#endif

#include <math.h>
#include <algorithm>

enum SortOrder {
    SORT_ASCENDING = 0,
    SORT_DESCENDING = 1
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Sorter
{
private:
    #ifdef __USE_ASL__
    static void inner_sort(int *_data, vgl_sort_indexes *_indexes, long long _size, SortOrder _sort_order)
    {
        cout << "insde ASL sort v2.0" << endl;
        ASL_CALL(asl_library_initialize());
        asl_sort_t hnd;
        if(_sort_order == SORT_ASCENDING)
        {
            ASL_CALL(asl_sort_create_i32(&hnd, ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO));
        }
        else if(_sort_order == SORT_DESCENDING)
        {
            ASL_CALL(asl_sort_create_i32(&hnd, ASL_SORTORDER_DESCENDING, ASL_SORTALGORITHM_AUTO));
        }

        // do sorting
        ASL_CALL(asl_sort_execute_i32(hnd, _size, _data, _indexes, _data, _indexes));

        ASL_CALL(asl_sort_destroy(hnd));
        ASL_CALL(asl_library_finalize());
    };
    #endif

    #ifndef __USE_ASL__
    static void inner_sort(int *_data, vgl_sort_indexes *_indexes, long long _size, SortOrder _sort_order)
    {
        int *work_buffer;
        MemoryAPI::allocate_array(&work_buffer, _size);

        if(_sort_order == SORT_ASCENDING)
        {
            stable_sort(_indexes, _indexes + _size, [&_data](long long _i1, long long _i2) {return _data[_i1] < _data[_i2];});
        }
        else if(_sort_order == SORT_DESCENDING)
        {
            stable_sort(_indexes, _indexes + _size, [&_data](long long _i1, long long _i2) {return _data[_i1] > _data[_i2];});
        }

        #pragma _NEC ivdep
        #pragma omp parallel for
        for(long long i = 0; i < _size; i++)
        {
            work_buffer[i] = _data[_indexes[i]];
        }

        #pragma _NEC ivdep
        #pragma omp parallel for
        for(long long i = 0; i < _size; i++)
        {
            _data[i] = work_buffer[i];
        }

        MemoryAPI::free_array(work_buffer);
    };
    #endif
public:
    static void sort(int *_data, vgl_sort_indexes *_indexes, long long _size, SortOrder _sort_order)
    {
        inner_sort(_data, _indexes, _size, _sort_order);
    };
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
