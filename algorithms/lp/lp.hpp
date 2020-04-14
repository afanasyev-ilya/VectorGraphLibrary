#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void LP::allocate_result_memory(int _vertices_count, int **_labels)
{
    MemoryAPI::allocate_array(_labels, _vertices_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void LP::free_result_memory(int *_labels)
{
    MemoryAPI::free_array(_labels);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void LP::performance_stats(string _name, double _time, long long _edges_count, int _iterations_count)
{
    cout << " --------------------------- " << _name << " performance stats --------------------------- " << endl;
    cout << "wall time: " << _time*1000.0 << " ms" << endl;
    cout << "wall perf: " << _edges_count / (_time * 1e6) << " MTEPS" << endl;
    cout << "iterations count: " << _iterations_count << endl;
    cout << "perf per iteration: " << _iterations_count * (_edges_count / (_time * 1e6)) << " MTEPS" << endl;
    cout << "band per iteration: " << INT_ELEMENTS_PER_EDGE * sizeof(int) * _iterations_count * (_edges_count / (_time * 1e9)) << " GB/s" << endl;
    cout << " ----------------------------------------------------------------------------------------- " << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
