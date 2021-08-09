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

#include <algorithm>
#include <functional>

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
        ASL_CALL(asl_sort_execute_i32(hnd, _size, _data, ASL_NULL, _data, _indexes));

        ASL_CALL(asl_sort_destroy(hnd));
    };
    #endif

    #ifndef __USE_ASL__
    static void inner_sort(int *_data, vgl_sort_indexes *_indexes, long long _size, SortOrder _sort_order)
    {
        if(_indexes != NULL)
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
        }
        else
        {
            if(_sort_order == SORT_ASCENDING)
                stable_sort(_data, _data + _size);
            else if(_sort_order == SORT_DESCENDING)
                stable_sort(_data, _data + _size, greater<int>());
        }
    };
    #endif
public:
    static void sort(int *_data, vgl_sort_indexes *_indexes, long long _size, SortOrder _sort_order)
    {
        Timer tm;
        tm.start();
        inner_sort(_data, _indexes, _size, _sort_order);
        tm.end();
        tm.print_time_stats("SORTING");
    };
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
