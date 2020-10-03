#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
int ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::renumber_vertex_id(int _id)
{
    return forward_conversion[_id];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::renumber_vertex_array(float *_input_array, float *_output_array)
{
    #pragma _NEC ivdep
    #pragma omp parallel for
    for(int i = 0; i < this->vertices_count; i++)
    {
        _output_array[i] = _input_array[forward_conversion[i]];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
