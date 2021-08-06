#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum COPY_IF_TYPE
{
    SAVE_ORDER = 0,
    DONT_SAVE_ORDER = 1
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline int vector_dense_copy_if(const int * __restrict__ _in_data,
                                int *_out_data,
                                int *_tmp_buffer,
                                const int _size,
                                const int _shift,
                                const COPY_IF_TYPE _output_order = SAVE_ORDER,
                                const int _threads_count = MAX_SX_AURORA_THREADS)
{
    int max_buffer_size = _size / (VECTOR_LENGTH * MAX_SX_AURORA_THREADS) + 1;

    int output_size = 0;
    int shifts_array[MAX_SX_AURORA_THREADS];

    #pragma omp parallel num_threads(MAX_SX_AURORA_THREADS) shared(output_size)
    {
        int tid = omp_get_thread_num();
        int *private_buffer = &_tmp_buffer[VECTOR_LENGTH * max_buffer_size * tid];

        int reg_ptrs[VECTOR_LENGTH];

        #pragma _NEC vreg(reg_ptrs)

        #pragma _NEC vector
        for (int i = 0; i < VECTOR_LENGTH; i++)
        {
            reg_ptrs[i] = 0;
        }

        // copy data to buffers
        #pragma omp for schedule(static, 8)
        for (int vec_start = 0; vec_start < _size; vec_start += VECTOR_LENGTH)
        {
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                int val = 0;

                if((vec_start + i) < _size)
                    val = _in_data[vec_start + i];

                if(val > 0)
                {
                    int dst_buffer_idx = reg_ptrs[i] + i * max_buffer_size;
                    private_buffer[dst_buffer_idx] = _shift + vec_start + i;
                    reg_ptrs[i]++;
                }
            }
        }

        // calculate sizes
        int dump_sizes[VECTOR_LENGTH];
        #pragma _NEC vector
        for (int i = 0; i < VECTOR_LENGTH; i++)
        {
            dump_sizes[i] = reg_ptrs[i];
        }
        int private_size = 0;
        #pragma _NEC vector
        for (int reg_pos = 0; reg_pos < VECTOR_LENGTH; reg_pos++)
        {
            private_size += dump_sizes[reg_pos];
        }

        // calculate output offsets
        shifts_array[tid] = private_size;
        #pragma omp barrier
        #pragma omp master
        {
            int cur_shift = 0;
            for(int i = 1; i < MAX_SX_AURORA_THREADS; i++)
            {
                shifts_array[i] += shifts_array[i - 1];
            }
            output_size = shifts_array[MAX_SX_AURORA_THREADS - 1];
            for(int i = (MAX_SX_AURORA_THREADS - 1); i >= 1; i--)
            {
                shifts_array[i] = shifts_array[i - 1];
            }
            shifts_array[0] = 0;
        }
        #pragma omp barrier
        int output_offset = shifts_array[tid];

        // save data to output array

        if(_output_order == DONT_SAVE_ORDER)
        {
            int current_pos = 0;
            for(int reg_pos = 0; reg_pos < VECTOR_LENGTH; reg_pos++)
            {
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                for (int i = 0; i < dump_sizes[reg_pos]; i++)
                {
                    int src_buffer_idx = i + reg_pos * max_buffer_size;
                    _out_data[output_offset + current_pos + i] = private_buffer[src_buffer_idx];
                }
                current_pos += dump_sizes[reg_pos];
            }
        }
        else if(_output_order == SAVE_ORDER)
        {
            int max_work = 0;
            #pragma _NEC vector
            for(int reg_pos = 0; reg_pos < VECTOR_LENGTH; reg_pos++)
            {
                if(reg_ptrs[reg_pos] > max_work)
                    max_work = reg_ptrs[reg_pos];
            }

            int min_work = dump_sizes[0];
            #pragma _NEC vector
            for(int reg_pos = 0; reg_pos < VECTOR_LENGTH; reg_pos++)
            {
                if(reg_ptrs[reg_pos] < min_work)
                    min_work = reg_ptrs[reg_pos];
            }

            // save large part
            int current_pos = 0;
            for(int work_pos = 0; work_pos < min_work; work_pos++)
            {
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int src_buffer_idx = work_pos + i * max_buffer_size;
                    _out_data[output_offset + current_pos + i] = private_buffer[src_buffer_idx];
                }
                current_pos += VECTOR_LENGTH;
            }

            // save reminder
            for(int work_pos = min_work; work_pos < max_work; work_pos++)
            {
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    if(work_pos < reg_ptrs[i])
                    {
                        int src_buffer_idx = work_pos + i * max_buffer_size;
                        _out_data[output_offset + current_pos] = private_buffer[src_buffer_idx];
                        current_pos++;
                    }
                }
            }

            // save reminder
            /*for(int reg_pos = 0; reg_pos < VECTOR_LENGTH; reg_pos++)
            {
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                for (int i = min_work; i < dump_sizes[reg_pos]; i++)
                {
                    int src_buffer_idx = i + reg_pos * max_buffer_size;
                    _out_data[output_offset + current_pos + i - min_work] = private_buffer[src_buffer_idx];
                }
                current_pos += dump_sizes[reg_pos] - min_work;
            }*/
        }
    }

    return output_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline int prefix_sum_copy_if(const int * __restrict__ _in_data,
                              int *_out_data,
                              int *_tmp_buffer,
                              const int _size)
{
    int *suma;

    #pragma omp parallel
    {
        const int ithread = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        #pragma omp single
        {
            suma = new int[nthreads+1];
            suma[0] = 0;
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < _size; i++)
        {
            if(_in_data[i] > 0)
                _tmp_buffer[i] = 1;
            else
                _tmp_buffer[i] = 0;
        }

        int sum = 0;
        #pragma omp for schedule(static)
        for (int i = 0; i < _size; i++)
        {
            sum += _tmp_buffer[i];
            _tmp_buffer[i] = sum;
        }
        suma[ithread+1] = sum;

        #pragma omp barrier
        int offset = 0;
        for(int i=0; i<(ithread+1); i++)
        {
            offset += suma[i];
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < _size; i++)
        {
            _tmp_buffer[i] += offset;
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < _size; i++)
        {
            if(_in_data[i] > 0)
            {
                _out_data[_tmp_buffer[i] - 1] = i;
            }
        }
    }

    int output_size = 0;
    for(int i = 0; i < omp_get_max_threads(); i++)
    {
        output_size += suma[i];
    }

    delete[] suma;

    return output_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cond>
inline int generic_dense_copy_if(Cond &&_cond,
                                 int *_out_data,
                                 int *_tmp_buffer,
                                 const int _size,
                                 const int _shift,
                                 const COPY_IF_TYPE _output_order = SAVE_ORDER,
                                 const int _threads_count = MAX_SX_AURORA_THREADS)
{
    int max_buffer_size = _size / (VECTOR_LENGTH * MAX_SX_AURORA_THREADS) + 1;

    int output_size = 0;
    int shifts_array[MAX_SX_AURORA_THREADS];

    #pragma omp parallel num_threads(MAX_SX_AURORA_THREADS) shared(output_size)
    {
        int tid = omp_get_thread_num();
        int *private_buffer = &_tmp_buffer[VECTOR_LENGTH * max_buffer_size * tid];

        int reg_ptrs[VECTOR_LENGTH];

        #pragma _NEC vreg(reg_ptrs)

        #pragma _NEC vector
        for (int i = 0; i < VECTOR_LENGTH; i++)
        {
            reg_ptrs[i] = 0;
        }

        // copy data to buffers
        #pragma omp for schedule(static, 8)
        for (int vec_start = 0; vec_start < _size; vec_start += VECTOR_LENGTH)
        {
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                if(_cond(vec_start + i) != -1)
                {
                    int dst_buffer_idx = reg_ptrs[i] + i * max_buffer_size;
                    private_buffer[dst_buffer_idx] = _shift + vec_start + i;
                    reg_ptrs[i]++;
                }
            }
        }

        // calculate sizes
        int dump_sizes[VECTOR_LENGTH];
        #pragma _NEC vector
        for (int i = 0; i < VECTOR_LENGTH; i++)
        {
            dump_sizes[i] = reg_ptrs[i];
        }
        int private_size = 0;
        #pragma _NEC vector
        for (int reg_pos = 0; reg_pos < VECTOR_LENGTH; reg_pos++)
        {
            private_size += dump_sizes[reg_pos];
        }

        // calculate output offsets
        shifts_array[tid] = private_size;
        #pragma omp barrier
        #pragma omp master
        {
            int cur_shift = 0;
            for(int i = 1; i < MAX_SX_AURORA_THREADS; i++)
            {
                shifts_array[i] += shifts_array[i - 1];
            }
            output_size = shifts_array[MAX_SX_AURORA_THREADS - 1];
            for(int i = (MAX_SX_AURORA_THREADS - 1); i >= 1; i--)
            {
                shifts_array[i] = shifts_array[i - 1];
            }
            shifts_array[0] = 0;
        }
        #pragma omp barrier
        int output_offset = shifts_array[tid];

        // save data to output array

        if(_output_order == DONT_SAVE_ORDER)
        {
            int current_pos = 0;
            for(int reg_pos = 0; reg_pos < VECTOR_LENGTH; reg_pos++)
            {
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                for (int i = 0; i < dump_sizes[reg_pos]; i++)
                {
                    int src_buffer_idx = i + reg_pos * max_buffer_size;
                    _out_data[output_offset + current_pos + i] = private_buffer[src_buffer_idx];
                }
                current_pos += dump_sizes[reg_pos];
            }
        }
        else if(_output_order == SAVE_ORDER)
        {
            int max_work = 0;
            #pragma _NEC vector
            for(int reg_pos = 0; reg_pos < VECTOR_LENGTH; reg_pos++)
            {
                if(reg_ptrs[reg_pos] > max_work)
                    max_work = reg_ptrs[reg_pos];
            }

            int min_work = dump_sizes[0];
            #pragma _NEC vector
            for(int reg_pos = 0; reg_pos < VECTOR_LENGTH; reg_pos++)
            {
                if(reg_ptrs[reg_pos] < min_work)
                    min_work = reg_ptrs[reg_pos];
            }

            // save large part
            int current_pos = 0;
            for(int work_pos = 0; work_pos < min_work; work_pos++)
            {
                #pragma _NEC ivdep
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int src_buffer_idx = work_pos + i * max_buffer_size;
                    _out_data[output_offset + current_pos + i] = private_buffer[src_buffer_idx];
                }
                current_pos += VECTOR_LENGTH;
            }

            // save reminder
            for(int work_pos = min_work; work_pos < max_work; work_pos++)
            {
                #pragma _NEC vovertake
                #pragma _NEC novob
                #pragma _NEC vector
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    if(work_pos < reg_ptrs[i])
                    {
                        int src_buffer_idx = work_pos + i * max_buffer_size;
                        _out_data[output_offset + current_pos] = private_buffer[src_buffer_idx];
                        current_pos++;
                    }
                }
            }
        }
    }

    return output_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename CopyCondition>
inline int vector_copy_if_indexes(CopyCondition &&_cond,
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
inline int omp_copy_if_indexes(CopyCondition &&_cond,
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
        int buffer_max_size = _size/nthreads;
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

template <typename CopyCondition>
inline int copy_if_indexes(CopyCondition &&_cond,
                           int *_out_data,
                           size_t _size,
                           int *_buffer,
                           const int _index_offset)
{
    #ifdef __USE_NEC_SX_AURORA__
    return vector_copy_if_indexes(_cond, _out_data, _size, _buffer, _index_offset);
    #elif __USE_MULTICORE__
    return omp_copy_if_indexes(_cond, _out_data, _size, _buffer, _index_offset);
    #else
    throw "Error in copy_if_indexes : unsupported architecture";
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
