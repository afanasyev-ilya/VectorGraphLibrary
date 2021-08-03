#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VGL_Frontier::VGL_Frontier(VGL_Graph &_graph, TraversalDirection _direction)
{
    object_type = FRONTIER;

    if(_graph.get_container_type() == VECTOR_CSR_GRAPH)
        frontier_representation = new FrontierVectorCSR(_graph, _direction);
    else if(_graph.get_container_type() == EDGES_LIST_GRAPH)
        frontier_representation = new FrontierGeneral(_graph, _direction);
    else
        throw "Error in VGL_Frontier::VGL_Frontier : unsupported graph container type";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VGL_Frontier::~VGL_Frontier()
{
    delete frontier_representation;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

