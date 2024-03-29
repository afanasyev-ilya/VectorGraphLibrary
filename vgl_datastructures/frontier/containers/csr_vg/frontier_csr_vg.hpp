#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierCSR_VG::FrontierCSR_VG(VGL_Graph &_graph, TraversalDirection _direction) : BaseFrontier(_graph, _direction)
{
    direction = _direction;
    graph_ptr = &_graph;
    class_type = CSR_VG_FRONTIER;
    init();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierCSR_VG::init()
{
    int vertices_count = graph_ptr->get_vertices_count();
    MemoryAPI::allocate_array(&flags, vertices_count);
    MemoryAPI::allocate_array(&ids, vertices_count);
    MemoryAPI::allocate_array(&work_buffer, vertices_count + VECTOR_LENGTH * omp_get_max_threads());

    // by default frontier is all active
    sparsity_type = ALL_ACTIVE_FRONTIER;
    this->size = vertices_count;

    if(graph_ptr->get_container_type() != CSR_VG_GRAPH)
    {
        throw "Error: incorrect graph container type in FrontierCSR_VG::init";
    }

    copy_vertex_group_info_from_graph();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierCSR_VG::~FrontierCSR_VG()
{
    MemoryAPI::free_array(flags);
    MemoryAPI::free_array(ids);
    MemoryAPI::free_array(work_buffer);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void FrontierCSR_VG::move_to_host()
{
    int vertices_count = graph_ptr->get_vertices_count();
    MemoryAPI::move_array_to_host(flags, vertices_count);
    MemoryAPI::move_array_to_host(ids, vertices_count);
    MemoryAPI::move_array_to_host(work_buffer, vertices_count);

    for(int i = 0; i < CSR_VERTEX_GROUPS_NUM; i++)
        vertex_groups[i].move_to_host();
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void FrontierCSR_VG::move_to_device()
{
    int vertices_count = graph_ptr->get_vertices_count();
    MemoryAPI::move_array_to_device(flags, vertices_count);
    MemoryAPI::move_array_to_device(ids, vertices_count);
    MemoryAPI::move_array_to_device(work_buffer, vertices_count);

    for(int i = 0; i < CSR_VERTEX_GROUPS_NUM; i++)
        vertex_groups[i].move_to_device();
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
