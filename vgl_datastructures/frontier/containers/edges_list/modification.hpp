#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierEdgesList::set_all_active()
{
    sparsity_type = ALL_ACTIVE_FRONTIER;
    this->size = graph_ptr->get_vertices_count();
    neighbours_count = graph_ptr->get_edges_count();

    #pragma omp parallel // dummy for performance evaluation
    {};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierEdgesList::add_vertex(int _src_id)
{
    this->ids[this->size] = _src_id;
    this->flags[_src_id] = 1;
    this->size++;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierEdgesList::add_group_of_vertices(int *_vertex_ids, int _number_of_vertices)
{
    for(int i = 0; i < _number_of_vertices; i++)
        add_vertex(_vertex_ids[i]);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
