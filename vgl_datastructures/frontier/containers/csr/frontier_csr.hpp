#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierCSR::FrontierCSR(VGL_Graph &_graph, TraversalDirection _direction) : BaseFrontier(_graph, _direction)
{
    direction = _direction;
    graph_ptr = &_graph;
    class_type = CSR_FRONTIER;
    init();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierCSR::init()
{
    int vertices_count = graph_ptr->get_vertices_count();
    MemoryAPI::allocate_array(&flags, vertices_count);
    MemoryAPI::allocate_array(&ids, vertices_count);
    MemoryAPI::allocate_array(&work_buffer, vertices_count + VECTOR_LENGTH * MAX_SX_AURORA_THREADS);

    // by default frontier is all active
    sparsity_type = ALL_ACTIVE_FRONTIER;
    this->size = vertices_count;

    if(graph_ptr->get_container_type() != CSR_GRAPH)
    {
        throw "Error: incorrect graph container type in FrontierCSR::init";
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierCSR::~FrontierCSR()
{
    MemoryAPI::free_array(flags);
    MemoryAPI::free_array(ids);
    MemoryAPI::free_array(work_buffer);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
