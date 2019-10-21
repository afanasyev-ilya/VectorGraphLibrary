//
//  vector_dictionary.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 14/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef vector_dictionary_h
#define vector_dictionary_h

#include <stdio.h>
#include <string.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class VectorDictionary
{
private:
    int *keys;
    int *values;
    int total_size;
    int current_sizes[VECTOR_LENGTH];
    int found[VECTOR_LENGTH];
    int max_keys[VECTOR_LENGTH];
    int max_values[VECTOR_LENGTH];
public:
    VectorDictionary(int _size);
    ~VectorDictionary();
    inline void increment_values(int *_keys_to_update, int *_inc_required);
    inline void find_max_values(int *_max_values, int *_current_labels);
    void clear();
    
    void print();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vector_dictionary.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* vector_dictionary_h */
