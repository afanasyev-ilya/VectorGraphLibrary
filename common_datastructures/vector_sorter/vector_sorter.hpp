//
//  vector_sorter.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 22/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef vector_sorter_hpp
#define vector_sorter_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VectorSorter::VectorSorter(int _size)
{
    max_size = _size;
    
    tmp_data = new int[max_size * VECTOR_LENGTH];
    
    #ifdef __USE_NEC_SX_AURORA__
    #pragma retain(tmp_data)
    #endif
    
    current_size = 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VectorSorter::~VectorSorter()
{
    delete []tmp_data;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectorSorter::initial_sort()
{
    for(int pair_pos = 0; pair_pos < current_size; pair_pos += 2)
    {
        #pragma unroll
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            int first = tmp_data[ACCESS_SORT_DATA(pair_pos, i, dim_size)];
            int second = tmp_data[ACCESS_SORT_DATA(pair_pos + 1, i, dim_size)];
            if(first > second)
            {
                tmp_data[ACCESS_SORT_DATA(pair_pos, i, dim_size)] = second;
                tmp_data[ACCESS_SORT_DATA(pair_pos + 1, i, dim_size)] = first;
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectorSorter::merge_sorted_arrays(int *_data, int _first_start, int *_result_data, int _second_start, int _size)
{
    int *L[VECTOR_LENGTH];
    int *R[VECTOR_LENGTH];
    int *result[VECTOR_LENGTH];
 
    int reg_L_pos[VECTOR_LENGTH]; // Initial index of first subarray
    int reg_R_pos[VECTOR_LENGTH]; // Initial index of second subarray
    int reg_res_pos[VECTOR_LENGTH]; // Initial index of merged subarray
    
    #pragma unroll
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        L[i] = &(_data[ACCESS_SORT_DATA(_first_start, i, dim_size)]);
        R[i] = &(_data[ACCESS_SORT_DATA(_second_start, i, dim_size)]);
        result[i] = &(_result_data[ACCESS_SORT_DATA(_first_start, i, dim_size)]);
        reg_L_pos[i] = 0;
        reg_R_pos[i] = 0;
        reg_res_pos[i] = 0;
    }
    
    for(int ctr = 0; ctr < _size * 2; ctr++)
    {
        #pragma unroll
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            int L_ind = reg_L_pos[i];
            int R_ind = reg_R_pos[i];
            int res_ind = reg_res_pos[i];
            
            if(L_ind < _size && R_ind < _size)
            {
                if (L[i][L_ind] <= R[i][R_ind])
                {
                    result[i][res_ind] = L[i][L_ind];
                    reg_L_pos[i]++;
                }
                else
                {
                    result[i][res_ind] = R[i][R_ind];
                    reg_R_pos[i]++;
                }
                reg_res_pos[i]++;
            }
            else if(L_ind < _size)
            {
                result[i][res_ind] = L[i][L_ind];
                reg_L_pos[i]++;
                reg_res_pos[i]++;
            }
            else if(R_ind < _size)
            {
                result[i][res_ind] = R[i][R_ind];
                reg_R_pos[i]++;
                reg_res_pos[i]++;
            }
        }
    }
    
    for(int ctr = 0; ctr < _size*2; ctr++)
    {
        #pragma unroll
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            L[i][ctr] = result[i][ctr];
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectorSorter::sort_data(int *_data, int _current_size)
{
    current_size = _current_size;
    dim_size = current_size;
    
    for(int pos = 0; pos < current_size; pos++)
    {
        #pragma unroll
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            tmp_data[ACCESS_SORT_DATA(pos, i, dim_size)] = _data[ACCESS_SORT_DATA(pos, i, dim_size)];
        }
    }
    
    initial_sort();
    
    int sorted_size = 2;
    while (sorted_size * 2 <= current_size)
    {
        for(int segment_start = 0; segment_start < current_size; segment_start += sorted_size * 2)
        {
            merge_sorted_arrays(tmp_data, segment_start, _data, segment_start + sorted_size, sorted_size);
        }
        sorted_size *= 2;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectorSorter::find_max(int *_most_frequent_labels, int *_current_labels)
{
    int last_pos[VECTOR_LENGTH];
    int max_size[VECTOR_LENGTH];
    int max_label[VECTOR_LENGTH];
    
    #pragma unroll
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        last_pos[i] = 0;
        max_label[i] = _current_labels[i];
        max_size[i] = 0;
    }
    
    for(int pos = 1; pos < current_size; pos++)
    {
        #pragma unroll
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            if(tmp_data[ACCESS_SORT_DATA(pos, i, dim_size)] != tmp_data[ACCESS_SORT_DATA(pos - 1, i, dim_size)])
            {
                int current_community_size = pos - last_pos[i];
                int current_community_label = tmp_data[ACCESS_SORT_DATA(pos - 1, i, dim_size)];
                last_pos[i] = pos;
                if((current_community_label != -1) && (max_size[i] < current_community_size))
                {
                    max_size[i] = current_community_size;
                    max_label[i] = current_community_label;
                }
            }
        }
    }
    
    #pragma unroll
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        int current_community_size = current_size - last_pos[i];
        int current_community_label = tmp_data[ACCESS_SORT_DATA(current_size - 1, i, dim_size)];
        
        if((current_community_label != -1) && (max_size[i] < current_community_size))
        {
            max_size[i] = current_community_size;
            max_label[i] = current_community_label;
        }
        
        _most_frequent_labels[i] = max_label[i];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectorSorter::clear()
{
    current_size = 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectorSorter::print()
{
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        for(int j = 0; j < current_size; j++)
        {
            cout << tmp_data[ACCESS_SORT_DATA(j, i, dim_size)] << " ";
        }
        cout << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* vector_sorter_hpp */
