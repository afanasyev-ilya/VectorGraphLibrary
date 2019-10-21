//
//  verify_results.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 07/09/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef verify_results_h
#define verify_results_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void verify_results(_T *_first_result, _T *_second_result, int _elements_count, int _error_count_print = 30)
{
    int error_count = 0;
    for(int i = 0; i < _elements_count; i++)
    {
        if(_first_result[i] != _second_result[i])
        {
            error_count++;
            if(error_count < _error_count_print)
            {
                cout << i << ": " << _first_result[i] << " " << _second_result[i] << endl;
            }
        }
    }
    cout << "error count: " << error_count << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* verify_results_h */
