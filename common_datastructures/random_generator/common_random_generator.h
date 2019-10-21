//
//  common_random_generator.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 16/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef common_random_generator_h
#define common_random_generator_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CommonRandomGenerator
{
private:
    
public:
    CommonRandomGenerator() {};
    
    template <typename _T>
    void generate_array_of_random_uniform_values(_T *_array, int _size, _T _max_val)
    {
        for(int i = 0; i < _size; i++)
            _array[i] = rand() % ((int)_max_val);
    }
    
    ~CommonRandomGenerator() {};
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
void CommonRandomGenerator::generate_array_of_random_uniform_values<float>(float *_array, int _size, float _max_val)
{
    for(int i = 0; i < _size; i++)
        _array[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/_max_val));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* common_random_generator_h */
