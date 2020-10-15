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
bool verify_results(VectCSRGraph &_graph,
                    VerticesArrayNEC<_T> &_first,
                    VerticesArrayNEC<_T> &_second,
                    int _first_printed_results = 0,
                    int _error_count_print = 30)
{
    // remember current directions for both arrays
    TraversalDirection prev_first_direction = _first.get_direction();
    TraversalDirection prev_second_direction = _second.get_direction();

    // make both results stored in original order
    _graph.reorder(_first, ORIGINAL);
    _graph.reorder(_second, ORIGINAL);

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

    if(error_count == 0)
        cout << "Results are equal" << endl;
    else
        cout << "Results are NOT equal, error_count = " << error_count << endl;

    if(error_count == 0)
        return true;
    else
        return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
bool equal_components(VectCSRGraph &_graph,
                      VerticesArrayNEC<_T> &_first,
                      VerticesArrayNEC<_T> &_second)
{
    int vertices_count = _graph.get_vertices_count();

    // remember current directions for both arrays
    TraversalDirection prev_first_direction = _first.get_direction();
    TraversalDirection prev_second_direction = _second.get_direction();

    // make both results stored in original order
    _graph.reorder(_first, ORIGINAL);
    _graph.reorder(_second, ORIGINAL);

    // construct equality maps
    map<int, int> f_s_equality;
    map<int, int> s_f_equality;
    for (int i = 0; i < vertices_count; i++)
    {
        f_s_equality[_first[i]] = _second[i];
        s_f_equality[_second[i]] = _first[i];
    }

    // check if components are equal using maps
    bool result = true;
    int error_count = 0;
    for (int i = 0; i < vertices_count; i++)
    {
        if (f_s_equality[_first[i]] != _second[i])
        {
            result = false;
            error_count++;
        }
        if (s_f_equality[_second[i]] != _first[i])
        {
            result = false;
            error_count++;
        }
    }

    // restore order if required
    _graph.reorder(_first, prev_first_direction);
    _graph.reorder(_second, prev_second_direction);

    if(result == true)
        cout << "Components are equal" << endl;
    else
        cout << "Components are NOT equal, error_count = " << error_count << endl;

    return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void print_component_sizes(VectCSRGraph &_graph, VerticesArrayNEC<_T> &_components)
{
    int vertices_count = _graph.get_vertices_count();

    // calculate sizes of each component
    map<int, int> components_sizes;
    for(int i = 0; i < vertices_count; i++)
    {
        int CC_num = _components[i];
        components_sizes[CC_num]++;
    }

    // calculate sizes stats
    map<int, int> sizes_stats;
    for(auto it = components_sizes.begin(); it != components_sizes.end(); ++it)
    {
        int CC_num = it->first;
        int CC_size = it->second;
        sizes_stats[CC_size]++;
    }

    // print sizes stats
    for(auto it = sizes_stats.begin(); it != sizes_stats.end(); ++it)
    {
        int size = it->first;
        int count = it->second;
        cout << "There are " << count << " components of size " << size << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

