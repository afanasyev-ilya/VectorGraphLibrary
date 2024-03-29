#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VGL_Frontier::VGL_Frontier(VGL_Graph &_graph, TraversalDirection _direction)
{
    object_type = FRONTIER;

    if(_graph.get_container_type() == VECTOR_CSR_GRAPH)
        container = new FrontierVectorCSR(_graph, _direction);
    else if(_graph.get_container_type() == EDGES_LIST_GRAPH)
        container = new FrontierEdgesList(_graph, _direction);
    else if(_graph.get_container_type() == CSR_GRAPH)
        container = new FrontierCSR(_graph, _direction);
    else if(_graph.get_container_type() == CSR_VG_GRAPH)
        container = new FrontierCSR_VG(_graph, _direction);
    else
        throw "Error in VGL_Frontier::VGL_Frontier : unsupported graph container type";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VGL_Frontier::VGL_Frontier(const VGL_Frontier &_copy_obj)
{
    throw "Error in VGL_Frontier::VGL_Frontier (copy constructor) : can not copy frontiers, please pass them by reference";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VGL_Frontier::~VGL_Frontier()
{
    delete container;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

