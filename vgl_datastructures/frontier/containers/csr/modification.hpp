#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierCSR::set_all_active()
{
    sparsity_type = ALL_ACTIVE_FRONTIER;
    this->size = graph_ptr->get_vertices_count();
    neighbours_count = graph_ptr->get_edges_count();

    #ifdef __USE_CSR_VERTEX_GROUPS__
    fill_vertex_group_data();
    #endif

    #pragma omp parallel // dummy for performance evaluation
    {};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierCSR::fill_vertex_group_data()
{
    #ifndef __USE_GPU__
    create_vertices_group_array(large_degree, 256, 2147483647);
    create_vertices_group_array(degree_128_256, 128, 256);
    create_vertices_group_array(degree_64_128, 64, 128);
    create_vertices_group_array(degree_32_64, 32, 64);
    create_vertices_group_array(degree_16_32, 16, 32);
    create_vertices_group_array(degree_8_16, 8, 16);
    create_vertices_group_array(degree_0_8, 0, 8);
    #else
    create_vertices_group_array(large_degree, 1024, 2147483647);
    create_vertices_group_array(degree_32_1024, 32, 1024);
    create_vertices_group_array(degree_16_32, 16, 32);
    create_vertices_group_array(degree_8_16, 8, 16);
    create_vertices_group_array(degree_4_8, 4, 8);
    create_vertices_group_array(degree_0_4, 0, 4);

    large_degree.move_to_device();
    degree_32_1024.move_to_device();
    degree_16_32.move_to_device();
    degree_8_16.move_to_device();
    degree_4_8.move_to_device();
    degree_0_4.move_to_device();
    #endif
}

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

void FrontierCSR::create_vertices_group_array(CSRVertexGroup &_group_data, int _bottom, int _top)
{
    int local_group_size = 0;
    long long local_group_neighbours = 0;

    int frontier_size = size;

    for(int i = 0; i < frontier_size; i++)
    {
        int src_id = i;
        if(this->sparsity_type != ALL_ACTIVE_FRONTIER)
            src_id = this->ids[i];

        int connections_count = this->graph_ptr->get_connections_count(src_id, this->direction);
        if((connections_count >= _bottom) && (connections_count < _top))
        {
            local_group_neighbours += connections_count;
            local_group_size++;
        }
    }

    _group_data.resize(local_group_size);
    _group_data.neighbours = local_group_neighbours;

    int pos = 0;
    for(int i = 0; i < frontier_size; i++)
    {
        int src_id = i;
        if(this->sparsity_type != ALL_ACTIVE_FRONTIER)
            src_id = this->ids[i];

        int connections_count = this->graph_ptr->get_connections_count(src_id, this->direction);
        if((connections_count >= _bottom) && (connections_count < _top))
        {
            _group_data.ids[pos] = src_id;
            pos++;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
