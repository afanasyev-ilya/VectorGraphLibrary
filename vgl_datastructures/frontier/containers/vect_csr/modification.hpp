#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierVectorCSR::set_all_active()
{
    sparsity_type = ALL_ACTIVE_FRONTIER;
    this->size = graph_ptr->get_vertices_count();
    neighbours_count = graph_ptr->get_edges_count();

    vector_engine_part_neighbours_count = 0; // TODO
    vector_core_part_neighbours_count = 0; // TODO
    collective_part_neighbours_count = graph_ptr->get_edges_count(); // TODO

    vector_engine_part_type = ALL_ACTIVE_FRONTIER;
    vector_core_part_type = ALL_ACTIVE_FRONTIER;
    collective_part_type = ALL_ACTIVE_FRONTIER;

    VectorCSRGraph *vect_csr_graph = (VectorCSRGraph *)graph_ptr->get_direction_data(direction);

    vector_engine_part_size = vect_csr_graph->get_vector_engine_threshold_vertex();
    vector_core_part_size = vect_csr_graph->get_vector_core_threshold_vertex() - vector_engine_part_size;
    collective_part_size = vect_csr_graph->get_vertices_count() - vector_engine_part_size - vector_core_part_size;

    #pragma omp parallel // dummy for performance evaluation
    {};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierVectorCSR::add_vertex(int src_id)
{
    if(this->size > 0)
    {
        throw "Error in FrontierVectorCSR::add_vertex: VGL can not add vertex to non-empty frontier";
    }

    VectorCSRGraph *vect_csr_graph = (VectorCSRGraph *)graph_ptr->get_direction_data(direction);

    const int ve_threshold = vect_csr_graph->get_vector_engine_threshold_vertex();
    const int vc_threshold = vect_csr_graph->get_vector_core_threshold_vertex();
    const int vertices_count = vect_csr_graph->get_vertices_count();

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
        vector_engine_part_neighbours_count = vect_csr_graph->get_vertex_pointers()[src_id + 1] -
                vect_csr_graph->get_vertex_pointers()[src_id];
    }
    if((src_id >= ve_threshold) && (src_id < vc_threshold))
    {
        vector_core_part_size = 1;
        vector_core_part_neighbours_count = vect_csr_graph->get_vertex_pointers()[src_id + 1] -
                vect_csr_graph->get_vertex_pointers()[src_id];
    }
    if((src_id >= vc_threshold) && (src_id < vertices_count))
    {
        collective_part_size = 1;
        collective_part_neighbours_count = vect_csr_graph->get_vertex_pointers()[src_id + 1] -
                vect_csr_graph->get_vertex_pointers()[src_id];
    }

    sparsity_type = SPARSE_FRONTIER;
    this->size = 1;

    #ifdef __USE_MPI__
    throw "Error: MPI thresholds calculation is not implemented in FrontierVectorCSR::add_vertex";
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierVectorCSR::add_group_of_vertices(int *_vertex_ids, int _number_of_vertices)
{
    if(this->size > 0)
    {
        throw "VGL ERROR: can not add vertices to non-empty frontier";
    }

    Sorter::sort(_vertex_ids, NULL, _number_of_vertices, SORT_ASCENDING);
    int max_size = graph_ptr->get_vertices_count();
    memset(flags, 0, sizeof(int)*max_size);

    // copy ids to frontier inner datastrcuture
    #pragma _NEC vector
    #pragma omp parallel for
    for(int idx = 0; idx < _number_of_vertices; idx++)
    {
        ids[idx] = _vertex_ids[idx];
        flags[ids[idx]] = IN_FRONTIER_FLAG;
    }
    this->size = _number_of_vertices;

    VectorCSRGraph *vect_csr_graph = (VectorCSRGraph *)graph_ptr->get_direction_data(direction);

    #pragma _NEC vector
    #pragma omp parallel for
    for(int idx = 0; idx < this->size; idx++)
    {
        const int current_id = ids[idx];
        const int next_id = ids[idx+1];

        int current_size = vect_csr_graph->get_connections_count(current_id);
        int next_size = 0;
        if(idx < (this->size - 1))
        {
            next_size = vect_csr_graph->get_connections_count(next_id);
        }

        if((this->size > VECTOR_ENGINE_THRESHOLD_VALUE) && (next_size <= VECTOR_ENGINE_THRESHOLD_VALUE))
        {
            vector_engine_part_size = idx + 1;
        }

        if((this->size > VECTOR_CORE_THRESHOLD_VALUE) && (next_size <= VECTOR_CORE_THRESHOLD_VALUE))
        {
            vector_core_part_size = idx + 1 - vector_engine_part_size;
        }
    }
    collective_part_size = this->size - vector_engine_part_size - vector_core_part_size;

    sparsity_type = SPARSE_FRONTIER;
    vector_engine_part_type = SPARSE_FRONTIER;
    vector_core_part_type = SPARSE_FRONTIER;
    collective_part_type = SPARSE_FRONTIER;

    #ifdef __USE_MPI__
    throw "Error: MPI thresholds calculation is not implemented in FrontierVectorCSR::add_group_of_vertices";
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
