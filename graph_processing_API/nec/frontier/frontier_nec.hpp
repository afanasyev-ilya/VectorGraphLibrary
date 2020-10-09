#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierNEC::FrontierNEC(VectCSRGraph &_graph, TraversalDirection _direction)
{
    max_size = _graph.get_vertices_count();
    MemoryAPI::allocate_array(&flags, max_size);
    MemoryAPI::allocate_array(&ids, max_size);
    MemoryAPI::allocate_array(&work_buffer, max_size + VECTOR_LENGTH * MAX_SX_AURORA_THREADS);

    // by default frontier is all active
    type = ALL_ACTIVE_FRONTIER;
    current_size = max_size;

    direction = _direction;
    graph_ptr = &_graph;

    #pragma omp parallel
    {}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierNEC::~FrontierNEC()
{
    MemoryAPI::free_array(flags);
    MemoryAPI::free_array(ids);
    MemoryAPI::free_array(work_buffer);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierNEC::print_frontier_info()
{
    #pragma omp master
    {
        UndirectedGraph *current_direction_graph = graph_ptr->get_direction_graph_ptr(direction);

        const int ve_threshold = current_direction_graph->get_vector_engine_threshold_vertex();
        const int vc_threshold = current_direction_graph->get_vector_core_threshold_vertex();
        const int vertices_count = current_direction_graph->get_vertices_count();

        string status;
        if(type == ALL_ACTIVE_FRONTIER)
            status = "all active";
        if(type == SPARSE_FRONTIER)
            status = "sparse";
        if(type == DENSE_FRONTIER)
            status = "dense";

        if(type != ALL_ACTIVE_FRONTIER)
        {
            cout << "frontier status: " << get_frontier_status_string(type) << ": " << get_frontier_status_string(vector_engine_part_type) << "/" <<
                 get_frontier_status_string(vector_core_part_type) << "/" << get_frontier_status_string(collective_part_type) << endl;
            cout << "frontier size: " << current_size << " from " << max_size << ", " <<
                 (100.0 * current_size) / max_size << " %" << endl;
            if(ve_threshold > 0)
                cout << 100.0 * vector_engine_part_size / (ve_threshold) << " % active first part" << endl;
            cout << 100.0 * vector_core_part_size / (vc_threshold - ve_threshold) << " % active second part" << endl;
            cout << 100.0 * collective_part_size / (vertices_count - vc_threshold) << " % active third part" << endl;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

