//
//  copy_if.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 05/11/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef copy_if_h
#define copy_if_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline int sparse_copy_if(const int *_in_data,
                          int *_out_data,
                          int *_tmp_buffer,
                          const int _buffer_size,
                          const int _start,
                          const int _end,
                          const int _threads_count = MAX_SX_AURORA_THREADS)
{
    int size = _end - _start;
    int elements_per_thread = (_buffer_size - 1)/_threads_count + 1;
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
        for(int vec_start = _start; vec_start < _end; vec_start += VECTOR_LENGTH)
        {
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = vec_start + i;
                if((src_id < _end) && (_in_data[src_id] > 0))
                {
                    _tmp_buffer[current_pointers_reg[i]] = src_id;
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
                    private_ptr[local_pos] = _tmp_buffer[last_pointers_reg[i]];
                    last_pointers_reg[i]++;
                    local_pos++;
                }
            }
        }
    }
    
    return elements_count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum COPY_IF_TYPE
{
    SAVE_ORDER = 0,
    DONT_SAVE_ORDER = 1
};

inline int dense_copy_if(const int * __restrict__ _in_data,
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


inline int dense_copy_if2(const int * __restrict__ _in_data,
                         int *_out_data,
                         int *_tmp_buffer,
                         const int _size)
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
                int val = -1;

                if((vec_start + i) < _size)
                    val = _in_data[vec_start + i];

                if(val > -1)
                {
                    int dst_buffer_idx = reg_ptrs[i] + i * max_buffer_size;
                    private_buffer[dst_buffer_idx] = val;
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

    return output_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* copy_if_h */
