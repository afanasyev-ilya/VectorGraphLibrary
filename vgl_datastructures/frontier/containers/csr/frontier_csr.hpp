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

    #ifdef __USE_CSR_VERTEX_GROUPS__
    copy_vertex_group_info_from_graph();
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierCSR::~FrontierCSR()
{
    MemoryAPI::free_array(flags);
    MemoryAPI::free_array(ids);
    MemoryAPI::free_array(work_buffer);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void FrontierCSR::move_to_host()
{
    int vertices_count = graph_ptr->get_vertices_count();
    MemoryAPI::move_array_to_host(flags, vertices_count);
    MemoryAPI::move_array_to_host(ids, vertices_count);
    MemoryAPI::move_array_to_host(work_buffer, vertices_count);

    #ifdef __USE_CSR_VERTEX_GROUPS__
    large_degree.move_to_host();
    degree_32_1024.move_to_host();
    degree_16_32.move_to_host();
    degree_8_16.move_to_host();
    degree_4_8.move_to_host();
    degree_0_4.move_to_host();
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void FrontierCSR::move_to_device()
{
    int vertices_count = graph_ptr->get_vertices_count();
    MemoryAPI::move_array_to_device(flags, vertices_count);
    MemoryAPI::move_array_to_device(ids, vertices_count);
    MemoryAPI::move_array_to_device(work_buffer, vertices_count);

    #ifdef __USE_CSR_VERTEX_GROUPS__
    large_degree.move_to_device();
    degree_32_1024.move_to_device();
    degree_16_32.move_to_device();
    degree_8_16.move_to_device();
    degree_4_8.move_to_device();
    degree_0_4.move_to_device();
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
