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
    set_all_constant(0);

    VectorCSRGraph *outgoing_container = (VectorCSRGraph *)this->graph_ptr->get_outgoing_data();
    VectorCSRGraph *incoming_container = (VectorCSRGraph *)this->graph_ptr->get_incoming_data();

    RandomGenerator rng_api;
    rng_api.generate_array_of_random_values<_T>(this->outgoing_edges, this->edges_count, _max_rand);
    outgoing_container->get_ve_ptr()->copy_array_from_csr_to_ve(this->outgoing_edges_ve, this->outgoing_edges);

    if(this->graph_ptr->get_number_of_directions() == BOTH_DIRECTIONS)
    {
        this->graph_ptr->copy_outgoing_to_incoming_edges(this->outgoing_edges, this->incoming_edges);
        incoming_container->get_ve_ptr()->copy_array_from_csr_to_ve(this->incoming_edges_ve, this->incoming_edges);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_VectorCSR<_T>::print()
{
    cout << "Edges Array (VectCSR)" << endl;
    cout << "outgoing: ";
    for(long long i = 0; i < edges_count_in_outgoing_csr; i++)
    {
        cout << this->outgoing_edges[i] << " ";
    }
    cout << endl;

    for(long long i = 0; i < edges_count_in_outgoing_ve; i++)
    {
        cout << this->outgoing_edges_ve[i] << " ";
    }
    cout << endl;

    if(this->graph_ptr->get_number_of_directions() == BOTH_DIRECTIONS)
    {
        cout << "incoming: ";
        for(long long i = 0; i < edges_count_in_incoming_csr; i++)
        {
            cout << this->incoming_edges[i] << " ";
        }
        cout << endl;

        for(long long i = 0; i < edges_count_in_incoming_ve; i++)
        {
            cout << this->incoming_edges_ve[i] << " ";
        }
        cout << endl << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_VectorCSR<_T>::attach_pointer(_T *_outer_data)
{
    this->edges_data = _outer_data;

    this->outgoing_edges = this->edges_data;
    this->outgoing_edges_ve = &(this->edges_data[edges_count_in_outgoing_csr]);

    if(this->graph_ptr->get_number_of_directions() == BOTH_DIRECTIONS)
    {
        this->incoming_edges = &(this->edges_data[edges_count_in_outgoing_csr + edges_count_in_outgoing_ve]);
        this->incoming_edges_ve = &(this->edges_data[edges_count_in_outgoing_csr + edges_count_in_outgoing_ve + edges_count_in_incoming_csr]);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
