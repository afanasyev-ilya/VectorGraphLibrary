#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VerticesArray<_T>::reorder(TraversalDirection _output_dir)
{
    if(graph_ptr->get_type() == VECT_CSR_GRAPH)
    {
        VectCSRGraph *vect_ptr = (VectCSRGraph *)graph_ptr;
        vect_ptr->reorder(*this, _output_dir);
    }
    if(graph_ptr->get_type() == SHARDED_CSR_GRAPH)
    {
        if((_output_dir != ORIGINAL) && (this->direction != ORIGINAL))
        {
            throw "Error in VerticesArray<_T>::reorder : SHARDED_CSR_GRAPH reorder wrong directions, can be ORIGINAL only";
        }
    }
    if(graph_ptr->get_type() == EDGES_LIST_GRAPH)
    {
        if((_output_dir != ORIGINAL) && (this->direction != ORIGINAL))
        {
            throw "Error in VerticesArray<_T>::reorder : EDGES_LIST_GRAPH reorder wrong directions, can be ORIGINAL only";
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
