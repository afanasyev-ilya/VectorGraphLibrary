#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VerticesArray<_T>::reorder(TraversalDirection _output_dir)
{
    graph_ptr->reorder(*this, _output_dir);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
