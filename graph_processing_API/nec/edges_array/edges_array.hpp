#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
template <typename _TVertexValue, typename _TEdgeWeight>
EdgesArrayNec<_T>::EdgesArrayNec(VectCSRGraph<_TVertexValue, _TEdgeWeight> &_graph)
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
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
EdgesArrayNec<_T>::~EdgesArrayNec()
{
    MemoryAPI::free_array(edges_data);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
