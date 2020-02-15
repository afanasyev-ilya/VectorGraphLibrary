#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
BFS<_TVertexValue, _TEdgeWeight>::BFS(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph):
frontier(_graph.get_vertices_count())
{

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
BFS<_TVertexValue, _TEdgeWeight>::~BFS()
{

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::allocate_result_memory(int _vertices_count, int **_levels)
{
    MemoryAPI::allocate_array(_levels, _vertices_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::free_result_memory(int *_levels)
{
    MemoryAPI::free_array(_levels);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::performance_stats(string _name, double _time, long long _edges_count, int _iterations_count)
{
    cout << " ---------------- " << _name << " performance stats -------------------- " << endl;
    cout << "wall time: " << _time*1000.0 << " ms" << endl;
    cout << "wall perf: " << _edges_count / (_time * 1e6) << " MTEPS" << endl;
    cout << "iterations count: " << _iterations_count << endl;
    cout << "perf per iteration: " << _iterations_count * (_edges_count / (_time * 1e6)) << " MTEPS" << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

