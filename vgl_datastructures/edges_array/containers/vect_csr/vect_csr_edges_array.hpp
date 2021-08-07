#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
EdgesArray_VectorCSR<_T>::EdgesArray_VectorCSR(VGL_Graph &_graph)
{
    this->graph_ptr = &_graph;
    this->edges_count = _graph.get_edges_count();

    VectorCSRGraph *outgoing_container = (VectorCSRGraph *)_graph.get_outgoing_data();
    VectorCSRGraph *incoming_container = (VectorCSRGraph *)_graph.get_incoming_data();

    edges_count_in_outgoing_csr = _graph.get_edges_count();

    edges_count_in_outgoing_ve = outgoing_container->get_edges_count_in_ve();
    if(_graph.get_number_of_directions() == BOTH_DIRECTIONS)
    {
        edges_count_in_incoming_csr = _graph.get_edges_count();
        edges_count_in_incoming_ve = incoming_container->get_edges_count_in_ve();
    }
    else
    {
        edges_count_in_incoming_csr = 0;
        edges_count_in_incoming_ve = 0;
    }

    this->total_array_size = edges_count_in_outgoing_csr + edges_count_in_outgoing_ve + edges_count_in_incoming_csr + edges_count_in_incoming_ve;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
EdgesArray_VectorCSR<_T>::~EdgesArray_VectorCSR()
{
    if(!this->is_copy)
    {
        MemoryAPI::free_array(this->edges_data);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_VectorCSR<_T>::set_all_constant(_T _const)
{
    MemoryAPI::set(this->edges_data, _const, this->total_array_size);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_VectorCSR<_T>::set_all_random(_T _max_rand)
{
    cout << "here " << endl;
    VectorCSRGraph *outgoing_container = (VectorCSRGraph *)this->graph_ptr->get_outgoing_data();
    VectorCSRGraph *incoming_container = (VectorCSRGraph *)this->graph_ptr->get_incoming_data();

    RandomGenerator rng_api;
    rng_api.generate_array_of_random_values<_T>(outgoing_edges, this->edges_count, _max_rand);
    cout << "here 2 " << endl;
    outgoing_container->get_ve_ptr()->copy_array_from_csr_to_ve(outgoing_edges_ve, outgoing_edges);

    cout << "here 3 " << endl;
    if(this->graph_ptr->get_number_of_directions() == BOTH_DIRECTIONS)
    {
        this->graph_ptr->copy_outgoing_to_incoming_edges(outgoing_edges, incoming_edges);
        cout << "here 4 " << endl;
        incoming_container->get_ve_ptr()->copy_array_from_csr_to_ve(incoming_edges_ve, incoming_edges);
    }
    cout << "here 5 " << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_VectorCSR<_T>::set(int _src_id, int _dst_id, _T _val, TraversalDirection _direction)
{
    // get correct pointer
    /*VGL_Graph *vect_ptr = (VGL_Graph *)this->graph_ptr;
    long long edges_count = this->graph_ptr->get_edges_count();

    // set into both CSR and VE for Advance API
    _T *target_csr_buffer, *target_ve_buffer;
    if(_direction == SCATTER)
    {
        target_csr_buffer = outgoing_csr_ptr;
        target_ve_buffer = outgoing_ve_ptr;
    }
    else if(_direction == GATHER)
    {
        target_csr_buffer = incoming_csr_ptr;
        target_ve_buffer = incoming_ve_ptr;
    }

    VectorCSRGraph *direction_graph_ptr = vect_ptr->get_direction_graph_ptr(_direction);

    long long csr_edge_pos = direction_graph_ptr->get_csr_edge_id(_src_id, _dst_id);
    target_csr_buffer[csr_edge_pos] = _val;

    long long ve_edge_pos = direction_graph_ptr->get_ve_edge_id(_src_id, _dst_id);
    target_ve_buffer[ve_edge_pos] = _val;*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
_T EdgesArray_VectorCSR<_T>::get(int _src_id, int _dst_id, TraversalDirection _direction)
{
    // get correct pointer
    /*VGL_Graph *vect_ptr = (VGL_Graph *)this->graph_ptr;
    long long edges_count = this->graph_ptr->get_edges_count();

    // always get from CSR since it's faster
    _T *answer_csr_buffer_ptr;
    _T *answer_ve_buffer_ptr;
    if(_direction == SCATTER)
    {
        answer_csr_buffer_ptr = outgoing_csr_ptr;
        answer_ve_buffer_ptr = outgoing_ve_ptr;
    }
    else if(_direction == GATHER)
    {
        answer_csr_buffer_ptr = incoming_csr_ptr;
        answer_ve_buffer_ptr = incoming_ve_ptr;
    }

    VectorCSRGraph *direction_graph_ptr = vect_ptr->get_direction_graph_ptr(_direction);

    long long csr_edge_pos = direction_graph_ptr->get_csr_edge_id(_src_id, _dst_id);
    long long ve_edge_pos = direction_graph_ptr->get_ve_edge_id(_src_id, _dst_id);

    return answer_csr_buffer_ptr[csr_edge_pos];*/
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_VectorCSR<_T>::print()
{
    cout << "Edges Array (VectCSR)" << endl;
    cout << "outgoing: ";
    for(long long i = 0; i < edges_count_in_outgoing_csr; i++)
    {
        cout << outgoing_edges[i] << " ";
    }
    cout << endl;

    for(long long i = 0; i < edges_count_in_outgoing_ve; i++)
    {
        cout << outgoing_edges_ve[i] << " ";
    }
    cout << endl;

    if(this->graph_ptr->get_number_of_directions() == BOTH_DIRECTIONS)
    {
        cout << "incoming: ";
        for(long long i = 0; i < edges_count_in_incoming_csr; i++)
        {
            cout << incoming_edges[i] << " ";
        }
        cout << endl;

        for(long long i = 0; i < edges_count_in_incoming_ve; i++)
        {
            cout << incoming_edges_ve[i] << " ";
        }
        cout << endl << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_VectorCSR<_T>::attach_pointer(_T *_outer_data)
{
    this->edges_data = _outer_data;

    outgoing_edges = this->edges_data;
    outgoing_edges_ve = &(this->edges_data[edges_count_in_outgoing_csr]);

    if(this->graph_ptr->get_number_of_directions() == BOTH_DIRECTIONS)
    {
        incoming_edges = &(this->edges_data[edges_count_in_outgoing_csr + edges_count_in_outgoing_ve]);
        incoming_edges_ve = &(this->edges_data[edges_count_in_outgoing_csr + edges_count_in_outgoing_ve + edges_count_in_incoming_csr]);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
