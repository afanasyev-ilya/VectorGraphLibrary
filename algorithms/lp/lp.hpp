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