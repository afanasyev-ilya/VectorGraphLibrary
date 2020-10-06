#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename _T>
EdgesArrayNec<_TVertexValue, _TEdgeWeight, _T>::EdgesArrayNec(VectCSRGraph<_TVertexValue, _TEdgeWeight> &_graph)
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

template <typename _TVertexValue, typename _TEdgeWeight, typename _T>
EdgesArrayNec<_TVertexValue, _TEdgeWeight, _T>::~EdgesArrayNec()
{
    MemoryAPI::free_array(edges_data);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename _T>
void EdgesArrayNec<_TVertexValue, _TEdgeWeight, _T>::set_all_constant(_T _const)
{
    #pragma _NEC ivdep
    #pragma omp parallel for
    for(long long i = 0; i < wall_array_size; i++)
    {
        edges_data[i] = _const;
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename _T>
void EdgesArrayNec<_TVertexValue, _TEdgeWeight, _T>::set_all_random(_T _max_rand)
{
    // init CSR parts
    RandomGenerationAPI rng_api;
    rng_api.generate_array_of_random_values<_T>(outgoing_csr_ptr, edges_count_in_outgoing_ve, _max_rand);
    rng_api.generate_array_of_random_values<_T>(incoming_csr_ptr, edges_count_in_incoming_ve, _max_rand);

    // copy data from CSR parts to VE parts
    graph_ptr->get_outgoing_graph_ptr()->get_ve_ptr()->copy_array_from_csr_to_ve(outgoing_ve_ptr, outgoing_csr_ptr);
    graph_ptr->get_incoming_graph_ptr()->get_ve_ptr()->copy_array_from_csr_to_ve(incoming_ve_ptr, incoming_csr_ptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
