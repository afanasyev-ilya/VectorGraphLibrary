#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
EdgesArrayNEC<_T>::EdgesArrayNEC(VectCSRGraph &_graph)
{
    edges_count = _graph.get_edges_count();
    edges_count_in_outgoing_ve = _graph.get_edges_count_in_outgoing_ve();
    edges_count_in_incoming_ve = _graph.get_edges_count_in_incoming_ve();
    wall_array_size = edges_count/*outgoing csr*/ + edges_count/*incoming csr*/ +
                      edges_count_in_outgoing_ve + edges_count_in_incoming_ve;

    MemoryAPI::allocate_array(&edges_data, wall_array_size);
    outgoing_csr_ptr = &edges_data[0];
    outgoing_ve_ptr = &edges_data[edges_count];
    incoming_csr_ptr = &edges_data[edges_count + edges_count_in_outgoing_ve];
    incoming_ve_ptr = &edges_data[edges_count + edges_count_in_outgoing_ve + edges_count];

    graph_ptr = &_graph;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
EdgesArrayNEC<_T>::~EdgesArrayNEC()
{
    MemoryAPI::free_array(edges_data);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArrayNEC<_T>::set_all_constant(_T _const)
{
    MemoryAPI::set(edges_data, _const, wall_array_size);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArrayNEC<_T>::set_all_random(_T _max_rand)
{
    // init CSR parts
    RandomGenerationAPI rng_api;
    rng_api.generate_array_of_random_values<_T>(outgoing_csr_ptr, edges_count, _max_rand);

    graph_ptr->reorder_edges_to_gather(incoming_csr_ptr, outgoing_csr_ptr);

    // copy data from CSR parts to VE parts
    graph_ptr->get_outgoing_graph_ptr()->get_ve_ptr()->copy_array_from_csr_to_ve(outgoing_ve_ptr, outgoing_csr_ptr);
    graph_ptr->get_incoming_graph_ptr()->get_ve_ptr()->copy_array_from_csr_to_ve(incoming_ve_ptr, incoming_csr_ptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArrayNEC<_T>::set(int _src_id, int _dst_id, _T _val, TraversalDirection _direction)
{
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

    UndirectedCSRGraph *direction_graph_ptr = graph_ptr->get_direction_graph_ptr(_direction);

    long long csr_edge_pos = direction_graph_ptr->get_csr_edge_id(_src_id, _dst_id);
    target_csr_buffer[csr_edge_pos] = _val;

    long long ve_edge_pos = direction_graph_ptr->get_ve_edge_id(_src_id, _dst_id);
    target_ve_buffer[ve_edge_pos] = _val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
_T EdgesArrayNEC<_T>::get(int _src_id, int _dst_id, TraversalDirection _direction)
{
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

    UndirectedCSRGraph *direction_graph_ptr = graph_ptr->get_direction_graph_ptr(_direction);

    long long csr_edge_pos = direction_graph_ptr->get_csr_edge_id(_src_id, _dst_id);
    long long ve_edge_pos = direction_graph_ptr->get_ve_edge_id(_src_id, _dst_id);

    return answer_csr_buffer_ptr[csr_edge_pos];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArrayNEC<_T>::print()
{
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
