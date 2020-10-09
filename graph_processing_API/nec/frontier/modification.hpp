#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierNEC::set_all_active()
{
    type = ALL_ACTIVE_FRONTIER;
    current_size = max_size;

    #pragma omp parallel // dummy for performance evaluation
    {};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierNEC::add_vertex(int src_id)
{
    if(current_size > 0)
    {
        throw "Error in FrontierNEC::add_vertex: VGL can not add vertex to non-empty frontier";
    }

    UndirectedGraph *current_direction_graph = graph_ptr->get_direction_graph_ptr(direction);
    const int ve_threshold = current_direction_graph->get_vector_engine_threshold_vertex();
    const int vc_threshold = current_direction_graph->get_vector_core_threshold_vertex();
    const int vertices_count = current_direction_graph->get_vertices_count();

    ids[0] = src_id;
    flags[src_id] = IN_FRONTIER_FLAG;

    vector_engine_part_type = SPARSE_FRONTIER;
    vector_core_part_type = SPARSE_FRONTIER;
    collective_part_type = SPARSE_FRONTIER;

    vector_engine_part_size = 0;
    vector_core_part_size = 0;
    collective_part_size = 0;

    if(src_id < ve_threshold)
    {
        vector_engine_part_size = 1;
    }
    if((src_id >= ve_threshold) && (src_id < vc_threshold))
    {
        vector_core_part_size = 1;
    }
    if((src_id >= vc_threshold) && (src_id < vertices_count))
    {
        collective_part_size = 1;
    }

    type = SPARSE_FRONTIER;
    current_size = 1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierNEC::add_vertices(int *_vertex_ids, int _number_of_vertices)
{
    UndirectedGraph *current_direction_graph = graph_ptr->get_direction_graph_ptr(direction);
    LOAD_UNDIRECTED_CSR_GRAPH_DATA((*current_direction_graph));

    /*if(current_size > 0)
    {
        throw "VGL ERROR: can not add vertices to non-empty frontier";
    }*/

    // sort input array
    std::sort(&_vertex_ids[0], &_vertex_ids[_number_of_vertices]);
    //memset(flags, 0, sizeof(int)*max_size);

    // copy ids to frontier inner datastrcuture
    #pragma _NEC vector
    #pragma omp parallel for
    for(int idx = 0; idx < _number_of_vertices; idx++)
    {
        ids[idx] = _vertex_ids[idx];
        flags[ids[idx]] = IN_FRONTIER_FLAG;
    }
    current_size = _number_of_vertices;

    #pragma _NEC vector
    #pragma omp parallel for
    for(int idx = 0; idx < current_size; idx++)
    {
        const int current_id = ids[idx];
        const int next_id = ids[idx+1];

        int current_size = vertex_pointers[current_id + 1] - vertex_pointers[current_id];;
        int next_size = 0;
        if(idx < (current_size - 1))
        {
            next_size = vertex_pointers[next_id + 1] - vertex_pointers[next_id];
        }

        if((current_size > VECTOR_ENGINE_THRESHOLD_VALUE) && (next_size <= VECTOR_ENGINE_THRESHOLD_VALUE))
        {
            vector_engine_part_size = idx + 1;
        }

        if((current_size > VECTOR_CORE_THRESHOLD_VALUE) && (next_size <= VECTOR_CORE_THRESHOLD_VALUE))
        {
            vector_core_part_size = idx + 1 - vector_engine_part_size;
        }
    }
    collective_part_size = current_size - vector_engine_part_size - vector_core_part_size;

    vector_engine_part_type = SPARSE_FRONTIER;
    vector_core_part_type = SPARSE_FRONTIER;
    collective_part_type = SPARSE_FRONTIER;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
