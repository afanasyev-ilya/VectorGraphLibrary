#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void ConnectedComponents::allocate_result_memory(int _vertices_count, int **_components)
{
    MemoryAPI::allocate_array(_components, _vertices_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void ConnectedComponents::free_result_memory(int *_components)
{
    MemoryAPI::free_array(_components);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
