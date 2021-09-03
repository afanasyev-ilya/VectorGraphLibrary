#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CSRVertexGroupCellC::CSRVertexGroupCellC()
{
    max_size = 1;
    size = 1;
    neighbours = 0;
    cell_c_size = 1;
    MemoryAPI::allocate_array(&ids, size);
    MemoryAPI::allocate_array(&cell_c_adjacent_ids, cell_c_size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRVertexGroupCellC::copy(CSRVertexGroupCellC & _other_group)
{
    this->size = _other_group.size;
    this->max_size = _other_group.size;
    this->neighbours = _other_group.neighbours;
    this->cell_c_size = _other_group.cell_c_size;
    this->min_connections = _other_group.min_connections;
    this->max_connections = _other_group.max_connections;
    this->resize(this->max_size);
    #ifndef __USE_GPU__
    MemoryAPI::copy(this->ids, _other_group.ids, this->size);
    // TODO
    #else
    cudaMemcpy(this->ids, _other_group.ids, this->size * sizeof(int), cudaMemcpyDeviceToDevice);
    // TODO
    #endif
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

void CSRVertexGroupCellC::add_vertex(int _src_id)
{
    ids[size] = _src_id;
    size++;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename CopyCond>
void CSRVertexGroupCellC::copy_data_if(CSRVertexGroupCellC & _full_group, CopyCond copy_cond,int *_buffer)
{
    this->size = ParallelPrimitives::copy_if_data(copy_cond, _full_group.ids, this->ids, _full_group.size, _buffer, _full_group.size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRVertexGroupCellC::resize(int _new_size)
{
    max_size = _new_size;
    size = _new_size;
    cell_c_size = size * max_connections; // TODO
    MemoryAPI::free_array(ids);
    MemoryAPI::free_array(cell_c_adjacent_ids);
    if (_new_size == 0)
    {
        MemoryAPI::allocate_array(&ids, 1);
        MemoryAPI::allocate_array(&cell_c_adjacent_ids, 1);
    }
    else
    {
        MemoryAPI::allocate_array(&ids, _new_size);
        MemoryAPI::allocate_array(&cell_c_adjacent_ids, cell_c_size);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRVertexGroupCellC::print_ids()
{
    cout << "vertex group info: ";
    for (int i = 0; i < size; i++)
        cout << ids[i] << " ";
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CSRVertexGroupCellC::~CSRVertexGroupCellC()
{
    MemoryAPI::free_array(ids);
    MemoryAPI::free_array(cell_c_adjacent_ids);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void CSRVertexGroupCellC::move_to_host()
{
    if(size > 0)
        MemoryAPI::move_array_to_host(ids, size);
    // TODO
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void CSRVertexGroupCellC::move_to_device()
{
    if(size > 0)
        MemoryAPI::move_array_to_device(ids, size);
    // TODO
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRVertexGroupCellC::import(CSR_VG_Graph *_graph, int _bottom, int _top)
{
    LOAD_CSR_GRAPH_DATA((*_graph));

    int local_group_size = 0;
    long long local_group_neighbours = 0;

    min_connections = _bottom;
    max_connections = _top;

    for(int src_id = 0; src_id < vertices_count; src_id++)
    {
        int connections_count = _graph->get_connections_count(src_id);
        if((connections_count >= _bottom) && (connections_count < _top))
        {
            local_group_neighbours += connections_count;
            local_group_size++;
        }
    }

    resize(local_group_size);
    neighbours = local_group_neighbours;

    int vertex_pos = 0;
    for(int src_id = 0; src_id < vertices_count; src_id++)
    {
        int connections_count = _graph->get_connections_count(src_id);
        if((connections_count >= _bottom) && (connections_count < _top))
        {
            this->ids[vertex_pos] = src_id;
            vertex_pos++;
        }
    }

    long long new_size = 0;
    for(int pos = 0; pos < size - 256; pos += 256)
    {
        int max = 0;
        for(int i = 0; i < 256; i++)
        {
            int src_id = this->ids[pos + i];
            int connections_count = _graph->get_connections_count(src_id);
            if(connections_count > max)
                max = connections_count;
        }
        new_size += max * 256;
    }
    cout << "neighbours vg: " << ((double)new_size)/neighbours << " , " << ((double)size*(_top))/neighbours << endl;

    for(int edge_pos = 0; edge_pos < max_connections; edge_pos++)
    {
        for(int i = 0; i < size; i++)
        {
            int src_id = ids[i];
            int connections_count = _graph->get_connections_count(src_id);
            if(edge_pos < connections_count)
                cell_c_adjacent_ids[i + edge_pos * size] = _graph->get_edge_dst(src_id, edge_pos);
            else
                cell_c_adjacent_ids[i + edge_pos * size] = -1;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

