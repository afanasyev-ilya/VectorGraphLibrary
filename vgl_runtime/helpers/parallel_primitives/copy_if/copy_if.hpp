#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename CopyCondition>
inline int ParallelPrimitives::vector_copy_if_indexes(CopyCondition &&_cond,
                                                      int *_out_data,
                                                      size_t _size,
                                                      int *_buffer,
                                                      const int _index_offset)
{
    int _threads_count = MAX_SX_AURORA_THREADS;
    int elements_per_thread = (_size - 1)/_threads_count + 1;
    int elements_per_vector = (elements_per_thread - 1)/VECTOR_LENGTH + 1;
    int shifts_array[MAX_SX_AURORA_THREADS];

    int elements_count = 0;
    #pragma omp parallel num_threads(_threads_count) shared(elements_count)
    {
        int tid = omp_get_thread_num();
        int start_pointers_reg[VECTOR_LENGTH];
        int current_pointers_reg[VECTOR_LENGTH];
        int last_pointers_reg[VECTOR_LENGTH];

        #pragma _NEC vreg(start_pointers_reg)
        #pragma _NEC vreg(current_pointers_reg)
        #pragma _NEC vreg(last_pointers_reg)

        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            start_pointers_reg[i] = tid * elements_per_thread + i * elements_per_vector;
            current_pointers_reg[i] = tid * elements_per_thread + i * elements_per_vector;
            last_pointers_reg[i] = tid * elements_per_thread + i * elements_per_vector;
        }

        #pragma omp for schedule(static)
        for(int vec_start = 0; vec_start < _size; vec_start += VECTOR_LENGTH)
        {
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = vec_start + i;
                int global_id = src_id + _index_offset;
                if((src_id < _size) && (_cond(global_id) > 0))
                {
                    _buffer[current_pointers_reg[i]] = global_id;
                    current_pointers_reg[i]++;
                }
            }
        }

        int max_difference = 0;
        int save_values_per_thread = 0;
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            int difference = current_pointers_reg[i] - start_pointers_reg[i];
            save_values_per_thread += difference;
            if(difference > max_difference)
                max_difference = difference;
        }

        shifts_array[tid] = save_values_per_thread;
        #pragma omp barrier

        #pragma omp master
        {
            int cur_shift = 0;
            for(int i = 1; i < _threads_count; i++)
            {
                shifts_array[i] += shifts_array[i - 1];
            }

            elements_count = shifts_array[_threads_count - 1];

            for(int i = (_threads_count - 1); i >= 1; i--)
            {
                shifts_array[i] = shifts_array[i - 1];
            }
            shifts_array[0] = 0;
        }

        #pragma omp barrier

        int tid_shift = shifts_array[tid];
        int *private_ptr = &(_out_data[tid_shift]);
        #pragma omp single
        {
            for(int i = 0; i < _threads_count; i++)
            {
                cout << shifts_array[i] << " ";
            }
            cout << endl;
        };

        int local_pos = 0;
        #pragma _NEC novector
        for(int pos = 0; pos < max_difference; pos++)
        {
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int loc_size = current_pointers_reg[i] - start_pointers_reg[i];

                if(pos < loc_size)
                {
                    private_ptr[local_pos] = _buffer[last_pointers_reg[i]];
                    last_pointers_reg[i]++;
                    local_pos++;
                }
            }
        }
    }

    return elements_count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename CopyCondition>
inline int ParallelPrimitives::omp_copy_if_indexes(CopyCondition &&_cond,
                                                   int *_out_data,
                                                   size_t _size,
                                                   int *_buffer,
                                                   const int _index_offset)
{
    int omp_work_group_size = omp_get_max_threads();

    const int max_threads = 400;
    int sum_array[max_threads];
    if(omp_work_group_size > max_threads)
        throw " Error in omp_copy_if_indexes : max_threads = 400 is too small for this architecture, please increase";

    #pragma omp parallel
    {
        const int ithread = omp_get_thread_num();
        const int nthreads = omp_work_group_size;

        int local_pos = 0;
        int buffer_max_size = (_size - 1)/nthreads + 1;
        int *local_buffer = &_buffer[ithread*buffer_max_size];

        #pragma omp single
        {
            sum_array[0] = 0;
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < _size; i++)
        {
            int global_src_id = i + _index_offset;
            if(_cond(global_src_id) > 0)
            {
                local_buffer[local_pos] = global_src_id;
                local_pos++;
            }
        }

        int local_size = local_pos;
        sum_array[ithread+1] = local_pos;

        #pragma omp barrier
        int offset = 0;
        for(int i=0; i<(ithread+1); i++)
        {
            offset += sum_array[i];
        }

        int *dst_ptr = &_out_data[offset];
        for (int i = 0; i < local_size; i++)
        {
            dst_ptr[i] = local_buffer[i];
        }
    }

    int output_size = 0;
    for(int i = 0; i < (omp_work_group_size + 1); i++)
    {
        output_size += sum_array[i];
    }

    return output_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename CopyCondition, typename _T>
inline int ParallelPrimitives::omp_copy_if_data(CopyCondition &&_cond,
                                                _T *_in_data,
                                                _T *_out_data,
                                                size_t _size,
                                                _T *_buffer)
{
    int omp_work_group_size = omp_get_max_threads();

    const int max_threads = 400;
    int sum_array[max_threads];
    if(omp_work_group_size > max_threads)
        throw " Error in omp_copy_if_indexes : max_threads = 400 is too small for this architecture, please increase";

    #pragma omp parallel
    {
        const int ithread = omp_get_thread_num();
        const int nthreads = omp_work_group_size;

        int local_pos = 0;
        int buffer_max_size = (_size - 1)/nthreads + 1;
        int *local_buffer = &_buffer[ithread*buffer_max_size];

        #pragma omp single
        {
            sum_array[0] = 0;
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < _size; i++)
        {
            int old_data = _in_data[i];
            if(_cond(old_data))
            {
                local_buffer[local_pos] = old_data;
                local_pos++;
            }
        }

        int local_size = local_pos;
        sum_array[ithread+1] = local_pos;

        #pragma omp barrier
        int offset = 0;
        for(int i=0; i<(ithread+1); i++)
        {
            offset += sum_array[i];
        }

        _T *dst_ptr = &_out_data[offset];
        for (int i = 0; i < local_size; i++)
        {
            dst_ptr[i] = local_buffer[i];
        }
    }

    int output_size = 0;
    for(int i = 0; i < (omp_work_group_size + 1); i++)
    {
        output_size += sum_array[i];
    }
    return output_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>     // std::cout
#include <algorithm>    // std::set_difference, std::sort
#include <vector>       // std::vector

bool compare(vector<int> &v1, vector<int> &v2, vector<int> &v)
{
    v.resize(v1.size());                    // 0  0  0  0  0  0  0  0  0  0
    std::vector<int>::iterator it;
    std::sort (v2.begin(),v2.end());   // 10 20 30 40 50

    std::sort (v1.begin(),v1.end());     //  5 10 15 20 25

    if(v1 != v2)
    {
        cout << "Error! vectors are different" << endl;
        it=std::set_difference (v1.begin(), v1.end(), v2.begin(), v2.end(), v.begin());
        //  5 15 25  0  0  0  0  0  0  0
        v.resize(it-v.begin());                      //  5 15 25

        std::cout << "The difference has " << (v.size()) << " elements:\n";
        for (it=v.begin(); it!=v.end(); ++it)
            std::cout << ' ' << *it;
        std::cout << '\n';
        return false;
    }
    return true;
}

template <typename CopyCondition>
inline int ParallelPrimitives::copy_if_indexes(CopyCondition &&_cond,
                                               int *_out_data,
                                               size_t _size,
                                               int *_buffer,
                                               const int _index_offset)
{
    int num_elements = 0;
    #ifdef __USE_NEC_SX_AURORA__
    vector<int> v1, v2;
    int omp_num_elements = omp_copy_if_indexes(_cond, _out_data, _size, _buffer, _index_offset);
    for(int i = 0; i < omp_num_elements; i++)
    {
        v1.push_back(_out_data[i]);
    }

    num_elements = vector_copy_if_indexes(_cond, _out_data, _size, _buffer, _index_offset);
    for(int i = 0; i < num_elements; i++)
    {
        v2.push_back(_out_data[i]);
    }
    vector<int>diff;
    if(!compare(v1, v2, diff))
    {
        vector<int> dub_list;
        for(int i = 1; i < v2.size();i++)
            if(v2[i] == v2[i - 1])
            {
                 dub_list.push_back(v2[i]);
                 cout << "dublicate in v2! "<< v2[i] << endl;
            }
        for(auto dub: dub_list)
        {
            for(int i = 0; i < num_elements; i++)
                if(_out_data[i] == dub)
                    cout << "dub " << dub << " at pos " << i << " / " << num_elements << " vs size " << _size << endl;
        }

    }
    #elif __USE_MULTICORE__
    num_elements = omp_copy_if_indexes(_cond, _out_data, _size, _buffer, _index_offset);
    #else
    throw "Error in copy_if_indexes : unsupported architecture";
    #endif
    return num_elements;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename CopyCondition, typename _T>
inline int ParallelPrimitives::copy_if_data(CopyCondition &&_cond,
                                            _T *_in_data,
                                            _T *_out_data,
                                            size_t _size,
                                            _T *_buffer)
{
    int num_elements = 0;
    #ifdef __USE_NEC_SX_AURORA__
    num_elements = omp_copy_if_data(_cond, _in_data, _out_data, _size, _buffer); // TODO vector version
    #elif __USE_MULTICORE__
    num_elements = omp_copy_if_data(_cond, _in_data, _out_data, _size, _buffer);
    #elif __USE_GPU__
    num_elements = thrust::copy_if(thrust::device, _in_data, _in_data + _size, _out_data, _cond) - _out_data;
    #else
    throw "Error in copy_if_indexes : unsupported architecture";
    #endif
    return num_elements;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

