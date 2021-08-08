#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
EdgesArray<_T>::EdgesArray(VGL_Graph &_graph)
{
    if(_graph.get_container_type() == VECTOR_CSR_GRAPH)
    {
        container = new EdgesArray_VectorCSR<_T>(_graph);
    }
    else if(_graph.get_container_type() == EDGES_LIST_GRAPH)
    {
        container = new EdgesArray_EL<_T>(_graph);
    }
    else if(_graph.get_container_type() == CSR_GRAPH)
    {
        container = new EdgesArray_CSR<_T>(_graph);
    }
    else
    {
        throw "Error in EdgesArray::EdgesArray : unsupported graph type";
    }

    MemoryAPI::allocate_array(&edges_data, container->get_total_array_size());

    container->attach_pointer(edges_data);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
EdgesArray<_T>::~EdgesArray()
{
    delete container;
    MemoryAPI::free_array(edges_data);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _T>
void EdgesArray<_T>::move_to_device()
{
    MemoryAPI::move_array_to_device(edges_data, container->get_total_array_size());
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _T>
void EdgesArray<_T>::move_to_host()
{
    MemoryAPI::move_array_to_device(edges_data, container->get_total_array_size());
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
