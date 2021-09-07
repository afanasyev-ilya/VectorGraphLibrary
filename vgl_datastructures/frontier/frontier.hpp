#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VGL_Frontier::VGL_Frontier(VGL_Graph &_graph, TraversalDirection _direction)
{
    object_type = FRONTIER;
    graph_ptr = &_graph;

    alloc_container(_direction);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VGL_Frontier::alloc_container(TraversalDirection _direction)
{
    if(graph_ptr->get_container_type() == VECTOR_CSR_GRAPH)
        container = new FrontierVectorCSR(*graph_ptr, _direction);
    else if(graph_ptr->get_container_type() == EDGES_LIST_GRAPH)
        container = new FrontierEdgesList(*graph_ptr, _direction);
    else if(graph_ptr->get_container_type() == CSR_GRAPH)
        container = new FrontierCSR(*graph_ptr, _direction);
    else if(graph_ptr->get_container_type() == CSR_VG_GRAPH)
        container = new FrontierCSR_VG(*graph_ptr, _direction);
    else
        throw "Error in VGL_Frontier::VGL_Frontier : unsupported graph container type";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VGL_Frontier::VGL_Frontier(const VGL_Frontier &_copy_obj)
{
    this->object_type = _copy_obj.object_type;
    this->graph_ptr = _copy_obj.graph_ptr;

    alloc_container(_copy_obj.get_direction());

    MemoryAPI::copy(this->container, _copy_obj.container, 1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VGL_Frontier::~VGL_Frontier()
{
    delete container;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

