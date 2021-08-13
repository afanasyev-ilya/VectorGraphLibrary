#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierCSR::set_all_active()
{
    sparsity_type = ALL_ACTIVE_FRONTIER;
    this->size = graph_ptr->get_vertices_count();
    neighbours_count = graph_ptr->get_edges_count();

    #ifdef __USE_CSR_VERTEX_GROUPS__
    copy_vertex_group_info_from_graph();
    #endif

    #pragma omp parallel // dummy for performance evaluation
    {};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_CSR_VERTEX_GROUPS__
void FrontierCSR::copy_vertex_group_info_from_graph()
{
    CSRGraph *csr_graph = (CSRGraph *)graph_ptr->get_direction_data(direction);
    for(int i = 0; i < CSR_VERTEX_GROUPS_NUM; i++)
        vertex_groups[i].copy(csr_graph->vertex_groups[i]);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_CSR_VERTEX_GROUPS__
template <typename CopyCond>
void FrontierCSR::copy_vertex_group_info_from_graph_cond(CopyCond copy_cond)
{
    CSRGraph *csr_graph = (CSRGraph *)graph_ptr->get_direction_data(direction);
    for(int i = 0; i < CSR_VERTEX_GROUPS_NUM; i++)
        vertex_groups[i].copy_data_if(csr_graph->vertex_groups[i], copy_cond, work_buffer);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_CSR_VERTEX_GROUPS__
int FrontierCSR::get_size_of_vertex_groups()
{
    int sum = 0;
    for(int i = 0; i < CSR_VERTEX_GROUPS_NUM; i++)
        sum += vertex_groups[i].size;
    return sum;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_CSR_VERTEX_GROUPS__
size_t FrontierCSR::get_neighbours_of_vertex_groups()
{
    size_t sum = 0;
    for(int i = 0; i < CSR_VERTEX_GROUPS_NUM; i++)
        sum += vertex_groups[i].neighbours;
    return sum;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierCSR::add_vertex(int _src_id)
{
    #ifdef __USE_CSR_VERTEX_GROUPS__
    for(int i = 0; i < CSR_VERTEX_GROUPS_NUM; i++)
    {
        if(vertex_groups[i].id_in_range(_src_id, graph_ptr->get_direction_data(direction)->get_connections_count(_src_id)))
        {
            vertex_groups[i].add_vertex(_src_id);
            break;
        }
    }
    #endif

    this->ids[this->size] = _src_id;
    this->flags[_src_id] = 1;
    this->size++;
    neighbours_count += graph_ptr->get_direction_data(direction)->get_connections_count(_src_id);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierCSR::clear()
{
    sparsity_type = SPARSE_FRONTIER;
    this->size = 0;
    neighbours_count = 0;

    #ifdef __USE_CSR_VERTEX_GROUPS__
    for(int i = 0; i < CSR_VERTEX_GROUPS_NUM; i++)
        vertex_groups[i].size = 0;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierCSR::add_group_of_vertices(int *_vertex_ids, int _number_of_vertices)
{
    for(int i = 0; i < _number_of_vertices; i++)
        add_vertex(_vertex_ids[i]);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
