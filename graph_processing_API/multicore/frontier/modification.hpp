#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierMulticore::set_all_active()
{
    type = ALL_ACTIVE_FRONTIER;
    current_size = max_size;
    neighbours_count = graph_ptr->get_edges_count();

    vector_engine_part_neighbours_count = 0;
    vector_core_part_neighbours_count = 0;
    collective_part_neighbours_count = graph_ptr->get_edges_count(); // TODO

    #pragma omp parallel // dummy for performance evaluation
    {};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierMulticore::add_vertex(int src_id)
{
    if(current_size > 0)
    {
        throw "Error in FrontierMulticore::add_vertex: VGL can not add vertex to non-empty frontier";
    }

    UndirectedVectCSRGraph *current_direction_graph;
    if(graph_ptr->get_type() == VECT_CSR_GRAPH)
    {
        VectCSRGraph *vect_csr_ptr = (VectCSRGraph*)graph_ptr;
        current_direction_graph = vect_csr_ptr->get_direction_graph_ptr(direction);
    }
    else
    {
        throw "Error in FrontierMulticore::add_vertex : unsupported graph type";
    }

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

    vector_engine_part_neighbours_count = 0;
    vector_core_part_neighbours_count = 0;
    collective_part_neighbours_count = 0;

    if(src_id < ve_threshold)
    {
        vector_engine_part_size = 1;
        vector_engine_part_neighbours_count = current_direction_graph->get_vertex_pointers()[src_id + 1] -
                                              current_direction_graph->get_vertex_pointers()[src_id];
    }
    if((src_id >= ve_threshold) && (src_id < vc_threshold))
    {
        vector_core_part_size = 1;
        vector_core_part_neighbours_count = current_direction_graph->get_vertex_pointers()[src_id + 1] -
                                            current_direction_graph->get_vertex_pointers()[src_id];
    }
    if((src_id >= vc_threshold) && (src_id < vertices_count))
    {
        collective_part_size = 1;
        collective_part_neighbours_count = current_direction_graph->get_vertex_pointers()[src_id + 1] -
                                           current_direction_graph->get_vertex_pointers()[src_id];
    }

    type = SPARSE_FRONTIER;
    current_size = 1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierMulticore::add_group_of_vertices(int *_vertex_ids, int _number_of_vertices)
{
    if(current_size > 0)
    {
        throw "VGL ERROR: can not add vertices to non-empty frontier";
    }

    Sorter::sort(_vertex_ids, NULL, _number_of_vertices, SORT_ASCENDING);
    memset(flags, 0, sizeof(int)*max_size);

    // copy ids to frontier inner datastrcuture
    #pragma _NEC vector
    #pragma omp parallel for
    for(int idx = 0; idx < _number_of_vertices; idx++)
    {
        ids[idx] = _vertex_ids[idx];
        flags[ids[idx]] = IN_FRONTIER_FLAG;
    }
    current_size = _number_of_vertices;

    VectCSRGraph *vect_csr_ptr = NULL;
    if(graph_ptr->get_type() == VECT_CSR_GRAPH)
    {
        vect_csr_ptr = (VectCSRGraph*)graph_ptr;
    }
    else
    {
        throw "Error in FrontierNEC::add_vertex : unsupported graph type";
    }

    #pragma _NEC vector
    #pragma omp parallel for
    for(int idx = 0; idx < current_size; idx++)
    {
        const int current_id = ids[idx];
        const int next_id = ids[idx+1];

        int current_size = vect_csr_ptr->get_connections_count(current_id, this->direction);
        int next_size = 0;
        if(idx < (current_size - 1))
        {
            next_size = vect_csr_ptr->get_connections_count(next_id, this->direction);
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

    type = SPARSE_FRONTIER;
    vector_engine_part_type = SPARSE_FRONTIER;
    vector_core_part_type = SPARSE_FRONTIER;
    collective_part_type = SPARSE_FRONTIER;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
