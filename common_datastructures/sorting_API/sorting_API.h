//
//  sorting_API.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 21/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef sorting_API_h
#define sorting_API_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_ASL__
#include <asl.h>
#endif

#include <math.h>
#include <algorithm>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class SortingAPI
{
public:
    SortingAPI() {};
    
    void sort_array(double *_array, int _size)
    {
        #ifdef __USE_ASL__
        double *wk = new double[_size];
        asl_int_t *iwk = new asl_int_t[1];
        ASL_dssta1(_array, _size, 3, wk, iwk);
        delete []iwk;
        delete []wk;
        #else
        sort(_array, _array + _size);
        #endif
    }
    
    ~SortingAPI() {};
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* sorting_API_h */
