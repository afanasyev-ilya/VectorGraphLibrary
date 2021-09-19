#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CSRVertexGroupCellC::CSRVertexGroupCellC()
{
    size = 1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool CSRVertexGroupCellC::id_in_range(int _src_id, int _connections_count)
{
    if ((_connections_count >= min_connections) && (_connections_count < max_connections))
        return true;
    else
        return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRVertexGroupCellC::print_ids()
{
    cout << "vertex group info: ";
    for (int i = 0; i < size; i++)
        cout << vertex_ids[i] << " ";
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CSRVertexGroupCellC::~CSRVertexGroupCellC()
{
    MemoryAPI::free_array(vertex_ids);
    MemoryAPI::free_array(vector_group_ptrs);
    MemoryAPI::free_array(vector_group_sizes);
    MemoryAPI::free_array(adjacent_ids);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRVertexGroupCellC::import(CSR_VG_Graph *_graph, int _bottom, int _top)
{
    LOAD_CSR_GRAPH_DATA((*_graph));

    int local_group_size = 0;
    long long local_group_neighbours = 0;

    min_connections = _bottom;
    max_connections = _top;

    // compute number of vertices and edges in vertex group
    for(int src_id = 0; src_id < vertices_count; src_id++)
    {
        int connections_count = _graph->get_connections_count(src_id);
        if((connections_count >= _bottom) && (connections_count < _top))
        {
            local_group_neighbours += connections_count;
            local_group_size++;
        }
    }

    size = local_group_size;
    cout << "size: " << size << endl;
    vector_segments_count = (size - 1) / VECTOR_LENGTH + 1;

    if(size == 0)
    {
        vector_segments_count = 0;
        edges_count_in_ve = 0;
        MemoryAPI::allocate_array(&vertex_ids, 1);
        MemoryAPI::allocate_array(&vector_group_ptrs, 1);
        MemoryAPI::allocate_array(&vector_group_sizes, 1);
        MemoryAPI::allocate_array(&adjacent_ids, 1);
    }
    else
    {
        MemoryAPI::allocate_array(&this->vertex_ids, size);

        // generate list of vertex group ids
        int vertex_pos = 0;
        for(int src_id = 0; src_id < vertices_count; src_id++)
        {
            int connections_count = _graph->get_connections_count(src_id);
            if((connections_count >= _bottom) && (connections_count < _top))
            {
                this->vertex_ids[vertex_pos] = src_id;
                vertex_pos++;
            }
        }

        edges_count_in_ve = 0;
        for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
        {
            int vec_start = cur_vector_segment * VECTOR_LENGTH;
            int cur_max_connections_count = 0;
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int pos = vec_start + i;
                if(pos < size)
                {
                    int src_id = this->vertex_ids[i];
                    int connections_count = _graph->get_connections_count(src_id);
                    if(cur_max_connections_count < connections_count)
                        cur_max_connections_count = connections_count;
                }
            }
            edges_count_in_ve += cur_max_connections_count * VECTOR_LENGTH;
        }
        MemoryAPI::allocate_array(&vector_group_ptrs, vector_segments_count);
        MemoryAPI::allocate_array(&vector_group_sizes, vector_segments_count);
        MemoryAPI::allocate_array(&adjacent_ids, edges_count_in_ve + VECTOR_LENGTH);

        long long current_edge = 0;
        for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
        {
            int vec_start = cur_vector_segment * VECTOR_LENGTH;
            int cur_max_connections_count = 0;
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int pos = vec_start + i;
                if(pos < size)
                {
                    int src_id = this->vertex_ids[i];
                    int connections_count = _graph->get_connections_count(src_id);
                    if(cur_max_connections_count < connections_count)
                        cur_max_connections_count = connections_count;
                }
            }

            vector_group_ptrs[cur_vector_segment] = current_edge;
            vector_group_sizes[cur_vector_segment] = cur_max_connections_count;

            for(int edge_pos = 0; edge_pos < cur_max_connections_count; edge_pos++)
            {
                #pragma _NEC ivdep
                #pragma _NEC vector
                for(int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int pos = vec_start + i;
                    if(pos < size)
                    {
                        int src_id = this->vertex_ids[i];
                        int connections_count = _graph->get_connections_count(src_id);
                        if((pos < size) && (edge_pos < connections_count))
                        {
                            adjacent_ids[current_edge + i] = _graph->get_edge_dst(src_id, edge_pos);
                        }
                        else
                        {
                            adjacent_ids[current_edge + i] = src_id;
                        }
                    }
                }
                current_edge += VECTOR_LENGTH;
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

