#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierCSR_VG::set_all_active()
{
    sparsity_type = ALL_ACTIVE_FRONTIER;
    this->size = graph_ptr->get_vertices_count();
    neighbours_count = graph_ptr->get_edges_count();

    copy_vertex_group_info_from_graph();

    #pragma omp parallel // dummy for performance evaluation
    {};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierCSR_VG::copy_vertex_group_info_from_graph()
{
    CSR_VG_Graph *csr_graph = (CSR_VG_Graph *)graph_ptr->get_direction_data(direction);
    for(int i = 0; i < CSR_VERTEX_GROUPS_NUM; i++)
        vertex_groups[i].copy(csr_graph->vertex_groups[i]);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename CopyCond>
void FrontierCSR_VG::copy_vertex_group_info_from_graph_cond(CopyCond copy_cond)
{
    CSR_VG_Graph *csr_graph = (CSR_VG_Graph *)graph_ptr->get_direction_data(direction);
    for(int i = 0; i < CSR_VERTEX_GROUPS_NUM; i++)
        vertex_groups[i].copy_data_if(csr_graph->vertex_groups[i], copy_cond, work_buffer);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int FrontierCSR_VG::get_size_of_vertex_groups()
{
    int sum = 0;
    for(int i = 0; i < CSR_VERTEX_GROUPS_NUM; i++)
        sum += vertex_groups[i].get_size();
    return sum;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t FrontierCSR_VG::get_neighbours_of_vertex_groups()
{
    size_t sum = 0;
    for(int i = 0; i < CSR_VERTEX_GROUPS_NUM; i++)
        sum += vertex_groups[i].get_neighbours();
    return sum;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierCSR_VG::add_vertex(int _src_id)
{
    for(int i = 0; i < CSR_VERTEX_GROUPS_NUM; i++)
    {
        if(vertex_groups[i].id_in_range(_src_id, graph_ptr->get_direction_data(direction)->get_connections_count(_src_id)))
        {
            vertex_groups[i].add_vertex(_src_id);
            break;
        }
    }

    this->ids[this->size] = _src_id;
    this->flags[_src_id] = 1;
    this->size++;
    neighbours_count += graph_ptr->get_direction_data(direction)->get_connections_count(_src_id);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierCSR_VG::clear()
{
    sparsity_type = SPARSE_FRONTIER;
    this->size = 0;
    neighbours_count = 0;

    for(int i = 0; i < CSR_VERTEX_GROUPS_NUM; i++)
        vertex_groups[i].clear();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierCSR_VG::add_group_of_vertices(int *_vertex_ids, int _number_of_vertices)
{
    for(int i = 0; i < _number_of_vertices; i++)
        add_vertex(_vertex_ids[i]);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
