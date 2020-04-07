#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <limits>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool are_same(float a, float b)
{
    return fabs(a - b) < 100.0*std::numeric_limits<float>::epsilon();
}

bool are_same(double a, double b)
{
    return fabs(a - b) < 100.0*std::numeric_limits<double>::epsilon();
}

bool are_same(int a, int b)
{
    return a == b;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void verify_results(_T *_first_result, _T *_second_result, int _elements_count, int _error_count_print = 30)
{
    int error_count = 0;
    for(int i = 0; i < _elements_count; i++)
    {
        if(!are_same(_first_result[i], _second_result[i]))
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
