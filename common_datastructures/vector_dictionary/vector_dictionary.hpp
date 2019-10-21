//
//  vector_dictionary.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 14/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef vector_dictionary_hpp
#define vector_dictionary_hpp

#include <iostream>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VectorDictionary::VectorDictionary(int _size)
{
    total_size = _size;
    keys = new int[total_size * VECTOR_LENGTH];
    values = new int[total_size * VECTOR_LENGTH];
    
    #ifdef __USE_NEC_SX_AURORA__
    #pragma retain(keys)
    #pragma retain(values)
    #endif
    
    #pragma unroll
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        current_sizes[i] = 0;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectorDictionary::increment_values(int *_keys_to_update, int *_inc_required)
{
    #pragma unroll
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        found[i] = 0;
    }
    
    int max_size = 0;
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        if(current_sizes[i] > max_size)
        {
            max_size = current_sizes[i];
        }
    }
    
    for(int pos = 0; pos < max_size; pos++)
    {
        int *step_keys = &keys[pos * VECTOR_LENGTH];
        int *step_values = &values[pos * VECTOR_LENGTH];
        
        #pragma unroll
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            if((pos < current_sizes[i]) && (_keys_to_update[i] == step_keys[i]) && _inc_required[i])
            {
                step_values[i] += 1;
                found[i] = 1;
            }
        }
        
        /*int found_sum = 0;
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            found_sum += found[i];
        }
        
        if(found_sum == VECTOR_LENGTH) // fast cancel if everything is found
        {
            break;
        }*/
    }
    
    #pragma unroll
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        int current_pos = current_sizes[i];
        if((found[i] == 0) && _inc_required[i]) // if not found
        {
            keys[current_pos * VECTOR_LENGTH + i] = _keys_to_update[i];
            values[current_pos * VECTOR_LENGTH + i] = 1;
            current_sizes[i] += 1;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectorDictionary::print()
{
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        cout << "dict data " << i << ": ";
        for(int j = 0; j < current_sizes[i]; j++)
        {
            cout << "(" << keys[j * VECTOR_LENGTH + i] << ", " << values[j * VECTOR_LENGTH + i] << ") ";
        }
        cout << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectorDictionary::find_max_values(int *_max_keys, int *_current_labels)
{
    int max_size = 0;
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        if(current_sizes[i] > max_size)
        {
            max_size = current_sizes[i];
        }
    }
    
    #pragma unroll
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        max_values[i] = 0;
        max_keys[i] = _current_labels[i];
    }
    
    for(int pos = 0; pos < max_size; pos++)
    {
        #pragma unroll
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            if((pos < current_sizes[i]) && (values[pos * VECTOR_LENGTH + i] > max_values[i]))
            {
                max_values[i] = values[pos * VECTOR_LENGTH + i];
                max_keys[i] = keys[pos * VECTOR_LENGTH + i];
            }
        }
    }
    
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        _max_keys[i] = max_keys[i];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectorDictionary::clear()
{
    #pragma unroll
    for(int i = 0; i < VECTOR_LENGTH; i++)
    {
        current_sizes[i] = 0;
    }
    
    memset(keys, 0, VECTOR_LENGTH*total_size);
    memset(values, 0, VECTOR_LENGTH*total_size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VectorDictionary::~VectorDictionary()
{
    if(keys != NULL)
        delete []keys;
    
    if(values != NULL)
        delete []values;
    
    keys = NULL;
    values = NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* vector_dictionary_hpp */
