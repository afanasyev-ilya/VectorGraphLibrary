#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
EdgesArray_VEC<_T>::EdgesArray_VEC(VectCSRGraph &_graph)
{
    long long edges_count = _graph.get_edges_count();
    edges_count_in_outgoing_ve = _graph.get_edges_count_in_outgoing_ve();
    edges_count_in_incoming_ve = _graph.get_edges_count_in_incoming_ve();
    this->total_array_size = edges_count/*outgoing csr*/ + edges_count/*incoming csr*/ +
                      edges_count_in_outgoing_ve + edges_count_in_incoming_ve;

    cout << "edges_count_in_outgoing_ve: " << edges_count_in_outgoing_ve << endl;
    cout << "edges_count_in_incoming_ve: " << edges_count_in_incoming_ve << endl;

    MemoryAPI::allocate_array(&this->edges_data, this->total_array_size);
    outgoing_csr_ptr = &this->edges_data[0];
    outgoing_ve_ptr = &this->edges_data[edges_count];
    incoming_csr_ptr = &this->edges_data[edges_count + edges_count_in_outgoing_ve];
    incoming_ve_ptr = &this->edges_data[edges_count + edges_count_in_outgoing_ve + edges_count];

    this->graph_ptr = &_graph;
    this->is_copy = false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
EdgesArray_VEC<_T>::EdgesArray_VEC(const EdgesArray_VEC<_T> &_copy_obj)
{
    this->graph_ptr = _copy_obj.graph_ptr;
    this->edges_data = _copy_obj.edges_data;

    outgoing_csr_ptr = _copy_obj.outgoing_csr_ptr;
    incoming_csr_ptr = _copy_obj.incoming_csr_ptr;

    outgoing_ve_ptr = _copy_obj.outgoing_ve_ptr;
    incoming_ve_ptr = _copy_obj.incoming_ve_ptr;

    edges_count_in_outgoing_ve = _copy_obj.edges_count_in_outgoing_ve;
    edges_count_in_incoming_ve = _copy_obj.edges_count_in_incoming_ve;
    this->total_array_size = _copy_obj.total_array_size;

    this->is_copy = true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
EdgesArray_VEC<_T>::~EdgesArray_VEC()
{
    if(!this->is_copy)
    {
        MemoryAPI::free_array(this->edges_data);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_VEC<_T>::set_all_constant(_T _const)
{
    MemoryAPI::set(this->edges_data, _const, this->total_array_size);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_VEC<_T>::set_all_random(_T _max_rand)
{
    // get correct pointer
    VectCSRGraph *vect_ptr = (VectCSRGraph *)this->graph_ptr;
    long long edges_count = this->graph_ptr->get_edges_count();

    // init CSR parts
    RandomGenerator rng_api;
    rng_api.generate_array_of_random_values<_T>(outgoing_csr_ptr, edges_count, _max_rand);

    vect_ptr->reorder_edges_scatter_to_gather(incoming_csr_ptr, outgoing_csr_ptr);

    // copy data from CSR parts to VE parts
    vect_ptr->get_outgoing_graph_ptr()->get_ve_ptr()->copy_array_from_csr_to_ve(outgoing_ve_ptr, outgoing_csr_ptr);
    vect_ptr->get_incoming_graph_ptr()->get_ve_ptr()->copy_array_from_csr_to_ve(incoming_ve_ptr, incoming_csr_ptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_VEC<_T>::set(int _src_id, int _dst_id, _T _val, TraversalDirection _direction)
{
    // get correct pointer
    VectCSRGraph *vect_ptr = (VectCSRGraph *)this->graph_ptr;
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

    UndirectedCSRGraph *direction_graph_ptr = vect_ptr->get_direction_graph_ptr(_direction);

    long long csr_edge_pos = direction_graph_ptr->get_csr_edge_id(_src_id, _dst_id);
    target_csr_buffer[csr_edge_pos] = _val;

    long long ve_edge_pos = direction_graph_ptr->get_ve_edge_id(_src_id, _dst_id);
    target_ve_buffer[ve_edge_pos] = _val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
_T EdgesArray_VEC<_T>::get(int _src_id, int _dst_id, TraversalDirection _direction)
{
    // get correct pointer
    VectCSRGraph *vect_ptr = (VectCSRGraph *)this->graph_ptr;
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

    UndirectedCSRGraph *direction_graph_ptr = vect_ptr->get_direction_graph_ptr(_direction);

    long long csr_edge_pos = direction_graph_ptr->get_csr_edge_id(_src_id, _dst_id);
    long long ve_edge_pos = direction_graph_ptr->get_ve_edge_id(_src_id, _dst_id);

    return answer_csr_buffer_ptr[csr_edge_pos];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_VEC<_T>::print()
{
    long long edges_count = this->graph_ptr->get_edges_count();
    cout << "Edges Array (VectCSR)" << endl;
    cout << "outgoing_csr_ptr: ";
    for(long long i = 0; i < edges_count; i++)
    {
        cout << outgoing_csr_ptr[i] << " ";
    }
    cout << endl;

    for(long long i = 0; i < edges_count_in_outgoing_ve; i++)
    {
        cout << outgoing_ve_ptr[i] << " ";
    }
    cout << endl;

    for(long long i = 0; i < edges_count; i++)
    {
        cout << incoming_csr_ptr[i] << " ";
    }
    cout << endl;

    for(long long i = 0; i < edges_count_in_incoming_ve; i++)
    {
        cout << incoming_ve_ptr[i] << " ";
    }
    cout << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_VEC<_T>::operator = (const EdgesArray_EL<_T> &_el_data)
{
    cout << "in assign" << endl;

    // get correct pointer
    VectCSRGraph *vect_ptr = (VectCSRGraph *)this->graph_ptr;
    long long edges_count = this->graph_ptr->get_edges_count();

    _T *el_data_ptr = _el_data.get_ptr();
    vect_ptr->reorder_edges_original_to_scatter(outgoing_csr_ptr, el_data_ptr);
    vect_ptr->reorder_edges_scatter_to_gather(incoming_csr_ptr, outgoing_csr_ptr);

    // copy data from CSR parts to VE parts
    vect_ptr->get_outgoing_graph_ptr()->get_ve_ptr()->copy_array_from_csr_to_ve(outgoing_ve_ptr, outgoing_csr_ptr);
    vect_ptr->get_incoming_graph_ptr()->get_ve_ptr()->copy_array_from_csr_to_ve(incoming_ve_ptr, incoming_csr_ptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template class EdgesArray_VEC<int>;
template class EdgesArray_VEC<float>;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
