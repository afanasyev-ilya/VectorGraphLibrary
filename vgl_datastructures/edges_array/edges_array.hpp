#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
EdgesArray<_T>::EdgesArray(VGL_Graph &_graph)
{
    graph_ptr = &_graph;
    init_container();

    MemoryAPI::allocate_array(&edges_data, container->get_total_array_size());

    container->attach_pointer(edges_data);
    is_copy = false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray<_T>::init_container()
{
    if(graph_ptr->get_container_type() == VECTOR_CSR_GRAPH)
    {
        container = new EdgesArray_VectorCSR<_T>(*graph_ptr);
    }
    else if(graph_ptr->get_container_type() == EDGES_LIST_GRAPH)
    {
        container = new EdgesArray_EL<_T>(*graph_ptr);
    }
    else if(graph_ptr->get_container_type() == CSR_GRAPH)
    {
        container = new EdgesArray_CSR<_T>(*graph_ptr);
    }
    else if(graph_ptr->get_container_type() == CSR_VG_GRAPH)
    {
        container = new EdgesArray_CSR<_T>(*graph_ptr);
    }
    else
    {
        throw "Error in EdgesArray::init container: unsupported graph type";
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray<_T>::copy_container(BaseEdgesArray<_T> *_other_container)
{
    if(graph_ptr->get_container_type() == VECTOR_CSR_GRAPH)
    {

    }
    else if(graph_ptr->get_container_type() == EDGES_LIST_GRAPH)
    {

    }
    else if(graph_ptr->get_container_type() == CSR_GRAPH)
    {

    }
    else if(graph_ptr->get_container_type() == CSR_VG_GRAPH)
    {

    }
    else
    {
        throw "Error in EdgesArray::copy container: unsupported graph type";
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
EdgesArray<_T>::EdgesArray(const EdgesArray<_T> &_copy_obj)
{
    this->object_type = _copy_obj.object_type;
    this->container = _copy_obj.container;
    this->edges_data = _copy_obj.edges_data;
    this->is_copy = true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
EdgesArray<_T> & EdgesArray<_T>::operator=(const EdgesArray<_T> & _other)
{
    throw "Error in EdgesArray<_T> & EdgesArray<_T>::operator= : not implemented yet";

    return *this;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
EdgesArray<_T>::~EdgesArray()
{
    if(!this->is_copy)
    {
        delete container;
        MemoryAPI::free_array(edges_data);
    }
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

template <typename _T>
template <typename MergeOperation>
void EdgesArray<_T>::finalize_advance(MergeOperation &&merge_operation)
{
    if(container->get_base_graph_ptr()->get_container_type() == VECTOR_CSR_GRAPH)
    {
        EdgesArray_VectorCSR<_T> *vcsr_container = (EdgesArray_VectorCSR<_T> *) container;
        vcsr_container->finalize_advance(merge_operation);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

