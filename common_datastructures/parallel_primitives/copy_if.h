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

template <typename Condition>
inline int sparse_copy_if(int *_out_data,
                          int *_tmp_buffer,
                          int _start,
                          int _end,
                          Condition condition_op,
                          int _threads_count = MAX_SX_AURORA_THREADS)
{
    int size = _end - _start;
    int elements_per_thread = size/_threads_count;
    int elements_per_vector = elements_per_thread/VECTOR_LENGTH;
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
                if(/*(src_id < _end) && */condition_op(src_id))
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

template <typename Condition>
inline int dense_copy_if(int *_out_data,
                         int _size,
                         Condition condition_op,
                         int _threads_count = MAX_SX_AURORA_THREADS)
{
    int shifts_array[MAX_SX_AURORA_THREADS];
    int pos = 0;
    int elements_count = 0;
    #pragma omp parallel shared(shifts_array, elements_count) num_threads(_threads_count)
    {
        int tid = omp_get_thread_num();
        int local_number_of_values = 0;
        
        #pragma omp for schedule(static)
        for(int src_id = 0; src_id < _size; src_id++)
        {
            if(condition_op(src_id))
            {
                local_number_of_values++;
            }
        }
        
        shifts_array[tid] = local_number_of_values;
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
        #pragma omp for schedule(static)
        for(int src_id = 0; src_id < _size; src_id++)
        {
            if(condition_op(src_id))
            {
                private_ptr[local_pos] = src_id;
                local_pos++;
            }
        }
    }
    
    return elements_count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* copy_if_h */
