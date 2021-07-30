#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierVectorCSR::FrontierVectorCSR(VGL_Graph &_graph, TraversalDirection _direction) : BaseFrontier(_graph, _direction)
{
    max_size = _graph.get_vertices_count();
    direction = _direction;
    graph_ptr = &_graph;
    init();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierVectorCSR::init()
{
    MemoryAPI::allocate_array(&flags, max_size);
    MemoryAPI::allocate_array(&ids, max_size);
    MemoryAPI::allocate_array(&work_buffer, max_size + VECTOR_LENGTH * MAX_SX_AURORA_THREADS);

    // by default frontier is all active
    type = ALL_ACTIVE_FRONTIER;
    current_size = max_size;

    if(graph_ptr->get_container_type() != VECTOR_CSR_GRAPH)
    {
        throw "Error: incorrect graph container type in FrontierVectorCSR";
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierVectorCSR::~FrontierVectorCSR()
{
    MemoryAPI::free_array(flags);
    MemoryAPI::free_array(ids);
    MemoryAPI::free_array(work_buffer);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
