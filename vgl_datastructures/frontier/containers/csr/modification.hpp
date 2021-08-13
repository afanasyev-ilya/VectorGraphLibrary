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
    #ifndef __USE_GPU__
    large_degree.copy(csr_graph->large_degree);
    degree_128_256.copy(csr_graph->degree_128_256);
    degree_64_128.copy(csr_graph->degree_64_128);
    degree_32_64.copy(csr_graph->degree_32_64);
    degree_16_32.copy(csr_graph->degree_16_32);
    degree_8_16.copy(csr_graph->degree_8_16);
    degree_0_8.copy(csr_graph->degree_0_8);
    #else
    large_degree.copy(csr_graph->large_degree);
    degree_32_1024.copy(csr_graph->degree_32_1024);
    degree_16_32.copy(csr_graph->degree_16_32);
    degree_8_16.copy(csr_graph->degree_8_16);
    degree_4_8.copy(csr_graph->degree_4_8);
    degree_0_4.copy(csr_graph->degree_0_4);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_CSR_VERTEX_GROUPS__
template <typename CopyCond>
void FrontierCSR::copy_vertex_group_info_from_graph_cond(CopyCond copy_cond)
{
    CSRGraph *csr_graph = (CSRGraph *)graph_ptr->get_direction_data(direction);
    #ifdef __USE_GPU__
    large_degree.copy_data_if(csr_graph->large_degree, copy_cond, work_buffer);
    degree_32_1024.copy_data_if(csr_graph->degree_32_1024, copy_cond, work_buffer);
    degree_16_32.copy_data_if(csr_graph->degree_16_32, copy_cond, work_buffer);
    degree_8_16.copy_data_if(csr_graph->degree_8_16, copy_cond, work_buffer);
    degree_4_8.copy_data_if(csr_graph->degree_4_8, copy_cond, work_buffer);
    degree_0_4.copy_data_if(csr_graph->degree_0_4, copy_cond, work_buffer);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_CSR_VERTEX_GROUPS__
int FrontierCSR::get_size_of_vertex_groups()
{
    #ifndef __USE_GPU__
    return large_degree.size + degree_128_256.size + degree_64_128.size + degree_32_64.size + degree_16_32.size +
            degree_8_16.size + degree_0_8.size;
    #else
    return large_degree.size + degree_32_1024.size + degree_16_32.size + degree_8_16.size + degree_4_8.size +
            degree_0_4.size;
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierCSR::add_vertex(int _src_id)
{
    this->ids[this->size] = _src_id;
    this->flags[_src_id] = 1;
    this->size++;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierCSR::add_group_of_vertices(int *_vertex_ids, int _number_of_vertices)
{
    for(int i = 0; i < _number_of_vertices; i++)
        add_vertex(_vertex_ids[i]);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
