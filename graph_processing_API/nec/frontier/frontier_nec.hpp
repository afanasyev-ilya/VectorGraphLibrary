#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierNEC::FrontierNEC(VectCSRGraph &_graph, TraversalDirection _direction)
{
    max_size = _graph.get_vertices_count();
    direction = _direction;
    graph_ptr = &_graph;
    init();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierNEC::FrontierNEC(ShardedCSRGraph &_graph, TraversalDirection _direction)
{
    max_size = _graph.get_vertices_count();
    direction = _direction;
    graph_ptr = &_graph;
    init();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierNEC::init()
{
    MemoryAPI::allocate_array(&flags, max_size);
    MemoryAPI::allocate_array(&ids, max_size);
    MemoryAPI::allocate_array(&work_buffer, max_size + VECTOR_LENGTH * MAX_SX_AURORA_THREADS);

    // by default frontier is all active
    type = ALL_ACTIVE_FRONTIER;
    current_size = max_size;

    #pragma omp parallel
    {};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierNEC::~FrontierNEC()
{
    MemoryAPI::free_array(flags);
    MemoryAPI::free_array(ids);
    MemoryAPI::free_array(work_buffer);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_MPI__
pair<int,int> FrontierNEC::get_vector_engine_mpi_thresholds()
{
    UndirectedCSRGraph *current_direction_graph;
    VectCSRGraph *vect_csr_ptr = (VectCSRGraph*)graph_ptr;
    current_direction_graph = vect_csr_ptr->get_direction_graph_ptr(direction);

    return current_direction_graph->get_mpi_thresholds(vgl_library_data.get_mpi_rank(), 0, vector_engine_part_size);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_MPI__
pair<int,int> FrontierNEC::get_vector_core_mpi_thresholds()
{
    UndirectedCSRGraph *current_direction_graph;
    VectCSRGraph *vect_csr_ptr = (VectCSRGraph*)graph_ptr;
    current_direction_graph = vect_csr_ptr->get_direction_graph_ptr(direction);

    return current_direction_graph->get_mpi_thresholds(vgl_library_data.get_mpi_rank(), vector_engine_part_size,
                                                       vector_engine_part_size + vector_core_part_size);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_MPI__
pair<int,int> FrontierNEC::get_collective_mpi_thresholds()
{
    UndirectedCSRGraph *current_direction_graph;
    VectCSRGraph *vect_csr_ptr = (VectCSRGraph*)graph_ptr;
    current_direction_graph = vect_csr_ptr->get_direction_graph_ptr(direction);

    return current_direction_graph->get_mpi_thresholds(vgl_library_data.get_mpi_rank(),
                                                       vector_engine_part_size + vector_core_part_size,
                                                       vector_engine_part_size + vector_core_part_size + collective_part_size);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
