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
    graph_ptr = NULL; //TODO graph type?
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

