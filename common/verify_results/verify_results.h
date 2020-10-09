#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <limits>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline bool are_same(float a, float b)
{
    return fabs(a - b) < 100.0*std::numeric_limits<float>::epsilon();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline bool are_same(double a, double b)
{
    return fabs(a - b) < 100.0*std::numeric_limits<double>::epsilon();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline bool are_same(int a, int b)
{
    return a == b;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void verify_results(VectCSRGraph &_graph,
                    VerticesArrayNec<_T> &_first,
                    VerticesArrayNec<_T> &_second,
                    int _first_printed_results = 0,
                    int _error_count_print = 30)
{
    // remember current directions for both arrays
    TraversalDirection prev_first_direction = _first.get_direction();
    TraversalDirection prev_second_direction = _second.get_direction();

    // make both results stored in original order
    _graph.reorder_to_original(_first);
    _graph.reorder_to_original(_second);

    // calculate error count
    int error_count = 0;
    int vertices_count = _graph.get_vertices_count();
    for(int i = 0; i < vertices_count; i++)
    {
        if(!are_same(_first[i], _second[i]))
        {
            error_count++;
            if(error_count < _error_count_print)
            {
                cout << i << ": " << _first[i] << " " << _second[i] << endl;
            }
        }
    }
    cout << "error count: " << error_count << endl << endl;

    // print first results
    if(_first_printed_results > 0)
    {
        cout << "first 10 results: " << endl;
        for(int i = 0; i < _first_printed_results; i++)
            cout << _first[i] << " & " << _second[i] << endl;
        cout << endl << endl;
    }

    // restore order if required
    _graph.reorder(_first, prev_first_direction);
    _graph.reorder(_second, prev_second_direction);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
