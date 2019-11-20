//
//  get_elements_count.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 05/11/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef get_elements_count_h
#define get_elements_count_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline int get_elements_count(int *_data, int _size, int _desired_value)
{
    int count = 0;
    #pragma _NEC vector
    #pragma omp parallel for reduction(+: count)
    for(int i = 0; i < _size; i++)
    {
        int val = 0;
        if(_data[i] == _desired_value)
            val = 1;
        count += val;
    }
    
    return count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* get_elements_count_h */
