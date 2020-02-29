//
//  random_generator.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 14/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef random_generation_API_h
#define random_generation_API_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ASL_random_generator.h"
#include "common_random_generator.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class RandomGenerationAPI
{
private:
    #ifdef __USE_ASL__
    ASLRandomGenerator ASL_rng;
    #endif
    CommonRandomGenerator common_rng;
public:
    RandomGenerationAPI() {};
    
    ~RandomGenerationAPI() {};
    
    template <typename _T>
    void generate_array_of_random_values(_T *_array, long long _size, _T _max_val)
    {
        #ifdef __USE_ASL__
        ASL_rng.generate_array_of_random_uniform_values<_T>(_array, _size, _max_val);
        #else
        common_rng.generate_array_of_random_uniform_values<_T>(_array, _size, _max_val);
        #endif
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* random_generation_API.h */
