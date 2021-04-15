#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <limits>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline bool are_same(float a, float b)
{
    float acceptable_difference = std::numeric_limits<float>::epsilon()*100.0;
    return fabs(a - b) <= acceptable_difference;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline bool are_same(double a, double b)
{
    double acceptable_difference = std::numeric_limits<double>::epsilon()*100.0;
    return fabs(a - b) <= acceptable_difference;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline bool are_same(int a, int b)
{
    return a == b;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
bool verify_results(VerticesArray<_T> &_first,
                    VerticesArray<_T> &_second,
                    int _first_printed_results = 0,
                    int _error_count_print = 30)
{
    // remember current directions for both arrays
    TraversalDirection prev_first_direction = _first.get_direction();
    TraversalDirection prev_second_direction = _second.get_direction();

    // check if sizes are the same
    if(_first.size() != _second.size())
    {
        cout << "Results are NOT equal, incorrect sizes";
        return false;
    }

    // make both results stored in original order
    _first.reorder(ORIGINAL);
    _second.reorder(ORIGINAL);

    // calculate error count
    int error_count = 0;
    int vertices_count = _first.size();
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
    cout << "error count: " << error_count << endl;

    // print first results
    if(_first_printed_results > 0)
    {
        cout << "first " << _first_printed_results << " results: " << endl;
        for(int i = 0; i < _first_printed_results; i++)
            cout << _first[i] << " & " << _second[i] << endl;
        cout << endl << endl;
    }

    // restore order if required
    _first.reorder(prev_first_direction);
    _second.reorder(prev_second_direction);

    if(error_count == 0)
        cout << "Results are equal" << endl;
    else
        cout << "Results are NOT equal, error_count = " << error_count << endl;
    cout << endl;

    if(error_count == 0)
        return true;
    else
        return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
bool verify_results(_T *_first,
                    _T *_second,
                    long long int _size,
                    int _first_printed_results = 0,
                    int _error_count_print = 30)
{
    // calculate error count
    int error_count = 0;
    for(int i = 0; i < _size; i++)
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
    cout << "error count: " << error_count << endl;

    // print first results
    if(_first_printed_results > 0)
    {
        cout << "first " << _first_printed_results << " results: " << endl;
        for(int i = 0; i < _first_printed_results; i++)
            cout << _first[i] << " & " << _second[i] << endl;
        cout << endl << endl;
    }

    if(error_count == 0)
        cout << "Results are equal" << endl;
    else
        cout << "Results are NOT equal, error_count = " << error_count << endl;
    cout << endl;

    if(error_count == 0)
        return true;
    else
        return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
bool equal_components(VerticesArray<_T> &_first,
                      VerticesArray<_T> &_second)
{
    // remember current directions for both arrays
    TraversalDirection prev_first_direction = _first.get_direction();
    TraversalDirection prev_second_direction = _second.get_direction();

    // check if sizes are the same
    if(_first.size() != _second.size())
    {
        cout << "Results are NOT equal, incorrect sizes";
        return false;
    }

    // make both results stored in original order
    _first.reorder(ORIGINAL);
    _second.reorder(ORIGINAL);

    // construct equality maps
    map<int, int> f_s_equality;
    map<int, int> s_f_equality;
    int vertices_count = _first.size();
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
    cout << "error count: " << error_count << endl;

    // restore order if required
    _first.reorder(prev_first_direction);
    _second.reorder(prev_second_direction);

    if(error_count == 0)
        cout << "Results are equal" << endl;
    else
        cout << "Results are NOT equal, error_count = " << error_count << endl;

    return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void print_component_sizes(VerticesArray<_T> &_components)
{
    int vertices_count = _components.size();

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

template <typename _T>
void verify_colors(VectCSRGraph &_graph, VerticesArray<_T> &_colors, int _error_count_print = 30)
{
    size_t error_count = 0;
    for(int src_id = 0; src_id < _graph.get_vertices_count(); src_id++)
    {
        int connections_count = _graph.get_outgoing_connections_count(src_id);

        int src_color = _colors[src_id];
        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            int dst_id = _graph.get_outgoing_edge_dst(src_id, edge_pos);
            int dst_color = _colors[dst_id];
            if((src_color == dst_color) && (src_id != dst_id))
            {
                if(error_count < 30)
                    cout << "(" << src_id << ": " << src_color << ", " << dst_id << ": " << dst_color << ")" << endl;
                error_count++;
            }
        }
    }

    cout << "error count: " << error_count << endl;

    if(error_count == 0)
        cout << "Results are equal" << endl;
    else
        cout << "Results are NOT equal, error_count = " << error_count << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
