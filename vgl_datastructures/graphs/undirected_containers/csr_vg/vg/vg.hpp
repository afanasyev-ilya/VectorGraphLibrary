#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CSRVertexGroup::CSRVertexGroup()
{
    max_size = 1;
    size = 1;
    neighbours = 0;
    MemoryAPI::allocate_array(&ids, size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRVertexGroup::copy(CSRVertexGroup & _other_group)
{
    this->size = _other_group.size;
    this->max_size = _other_group.size;
    this->neighbours = _other_group.neighbours;
    this->resize(this->max_size);
    this->min_connections = _other_group.min_connections;
    this->max_connections = _other_group.max_connections;
    #ifndef __USE_GPU__
    MemoryAPI::copy(this->ids, _other_group.ids, this->size);
    #else
    cudaMemcpy(this->ids, _other_group.ids, this->size * sizeof(int), cudaMemcpyDeviceToDevice);
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool CSRVertexGroup::id_in_range(int _src_id, int _connections_count)
{
    if ((_connections_count >= min_connections) && (_connections_count < max_connections))
        return true;
    else
        return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRVertexGroup::add_vertex(int _src_id)
{
    ids[size] = _src_id;
    size++;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename CopyCond>
void CSRVertexGroup::copy_data_if(CSRVertexGroup & _full_group, CopyCond copy_cond,int *_buffer)
{
    this->size = ParallelPrimitives::copy_if_data(copy_cond, _full_group.ids, this->ids, _full_group.size,
                                                  _buffer, _full_group.size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRVertexGroup::resize(int _new_size)
{
    max_size = _new_size;
    size = _new_size;
    MemoryAPI::free_array(ids);
    if (_new_size == 0)
        MemoryAPI::allocate_array(&ids, 1);
    else
        MemoryAPI::allocate_array(&ids, _new_size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRVertexGroup::print_ids()
{
    cout << "vertex group info: ";
    for (int i = 0; i < size; i++)
        cout << ids[i] << " ";
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CSRVertexGroup::~CSRVertexGroup()
{
    MemoryAPI::free_array(ids);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void CSRVertexGroup::move_to_host()
{
    if(size > 0)
        MemoryAPI::move_array_to_host(ids, size);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void CSRVertexGroup::move_to_device()
{
    if(size > 0)
        MemoryAPI::move_array_to_device(ids, size);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRVertexGroup::import(CSR_VG_Graph *_graph, int _bottom, int _top)
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
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

