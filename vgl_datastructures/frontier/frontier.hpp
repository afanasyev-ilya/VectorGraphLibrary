#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VGL_Frontier::VGL_Frontier(VGL_Graph &_graph, TraversalDirection _direction)
{
    object_type = FRONTIER;
    direction = _direction;
    graph_ptr = &_graph;

    alloc_container();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VGL_Frontier::alloc_container()
{
    if(graph_ptr->get_container_type() == VECTOR_CSR_GRAPH)
        container = new FrontierVectorCSR(*graph_ptr, direction);
    else if(graph_ptr->get_container_type() == EDGES_LIST_GRAPH)
        container = new FrontierEdgesList(*graph_ptr, direction);
    else if(graph_ptr->get_container_type() == CSR_GRAPH)
        container = new FrontierCSR(*graph_ptr, direction);
    else if(graph_ptr->get_container_type() == CSR_VG_GRAPH)
        container = new FrontierCSR_VG(*graph_ptr, direction);
    else
        throw "Error in VGL_Frontier::VGL_Frontier : unsupported graph container type";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VGL_Frontier::VGL_Frontier(const VGL_Frontier &_copy_obj)
{
    this->object_type = _copy_obj.object_type;
    this->graph_ptr = _copy_obj.graph_ptr;
    this->direction = _copy_obj.direction;

    alloc_container();

    MemoryAPI::copy(this->container, _copy_obj.container, 1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VGL_Frontier::~VGL_Frontier()
{
    delete container;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

