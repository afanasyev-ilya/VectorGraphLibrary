/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VGL_Graph::reorder(VerticesArray<_T> &_data, TraversalDirection _output_dir)
{
    /*if(_output_dir == SCATTER)
    {
        reorder_to_scatter(_data);
    }
    if(_output_dir == GATHER)
    {
        reorder_to_gather(_data);
    }
    if(_output_dir == ORIGINAL)
    {
        reorder_to_original(_data);
    }*/
    cout << " VGL graph doing reorder " << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int VGL_Graph::reorder(int _vertex_id, TraversalDirection _input_dir, TraversalDirection _output_dir)
{
    if((_vertex_id < 0) || (_vertex_id >= this->vertices_count))
        throw "Error in VectCSRGraph::reorder : _vertex_id is out of range";

    if(_input_dir == ORIGINAL)
    {
        if(_output_dir == GATHER)
        {
            return incoming_data->reorder_to_sorted(_vertex_id);
        }
        if(_output_dir == SCATTER)
        {
            return outgoing_data->reorder_to_sorted(_vertex_id);
        }
    }
    if(_input_dir == SCATTER)
    {
        if(_output_dir == ORIGINAL)
        {
            return outgoing_data->reorder_to_original(_vertex_id);
        }
        if(_output_dir == GATHER)
        {
            return incoming_data->reorder_to_sorted(outgoing_data->reorder_to_original(_vertex_id));
        }
    }
    if(_input_dir == GATHER)
    {
        if(_output_dir == ORIGINAL)
        {
            return incoming_data->reorder_to_original(_vertex_id);
        }
        if(_output_dir == SCATTER)
        {
            return outgoing_data->reorder_to_sorted(incoming_data->reorder_to_original(_vertex_id));
        }
    }
    return -1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int VGL_Graph::select_random_vertex(TraversalDirection _direction)
{
    return reorder(outgoing_data->select_random_vertex(), SCATTER, _direction);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
