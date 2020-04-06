#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void PR::allocate_result_memory(int _vertices_count, float **_page_ranks)
{
    MemoryAPI::allocate_array(_page_ranks, _vertices_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void PR::free_result_memory(float *_page_ranks)
{
    MemoryAPI::free_array(_page_ranks);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void PR::performance_stats(string _name, double _time, long long _edges_count, int _iterations_count)
{
    cout << " --------------------------- " << _name << " performance stats --------------------------- " << endl;
    cout << "wall time: " << _time*1000.0 << " ms" << endl;
    cout << "wall perf: " << _edges_count / (_time * 1e6) << " MTEPS" << endl;
    cout << "iterations count: " << _iterations_count << endl;
    cout << "perf per iteration: " << _iterations_count * (_edges_count / (_time * 1e6)) << " MTEPS" << endl;
    cout << "band per iteration: " << 5.0 * sizeof(int) * _iterations_count * (_edges_count / (_time * 1e9)) << " GB/s" << endl;
    cout << " ----------------------------------------------------------------------------------------- " << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
