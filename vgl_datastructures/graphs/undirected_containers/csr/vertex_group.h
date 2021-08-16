#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct CSRVertexGroup
{
    int *ids;
    int size;
    int max_size;
    long long neighbours;

    int min_connections, max_connections;

    CSRVertexGroup()
    {
        max_size = 1;
        size = 1;
        neighbours = 0;
        MemoryAPI::allocate_array(&ids, size);
    }

    void copy(CSRVertexGroup &_other_group)
    {
        this->size = _other_group.size;
        this->max_size = _other_group.size;
        this->neighbours = _other_group.size;
        this->resize(this->max_size);
        this->min_connections = _other_group.min_connections;
        this->max_connections = _other_group.max_connections;
        #ifndef __USE_GPU__
        MemoryAPI::copy(this->ids, _other_group.ids, this->size);
        #else
        cudaMemcpy(this->ids, _other_group.ids, this->size * sizeof(int), cudaMemcpyDeviceToDevice);
        #endif
    }

    bool id_in_range(int _src_id, int _connections_count)
    {
        if((_connections_count >= min_connections) && (_connections_count < max_connections))
            return true;
        else
            return false;
    }

    void add_vertex(int _src_id)
    {
        ids[size] = _src_id;
        size++;
    }

    template <typename CopyCond>
    void copy_data_if(CSRVertexGroup &_full_group, CopyCond copy_cond, int *_buffer)
    {
        this->size = ParallelPrimitives::copy_if_data(copy_cond, _full_group.ids, this->ids, _full_group.size, _buffer);
    }

    void resize(int _new_size)
    {
        max_size = _new_size;
        size = _new_size;
        MemoryAPI::free_array(ids);
        if(_new_size == 1)
            MemoryAPI::allocate_array(&ids, 1);
        else
            MemoryAPI::allocate_array(&ids, _new_size);
    }

    void print_ids()
    {
        cout << "vertex group info: ";
        for(int i = 0; i < size; i++)
            cout << ids[i] << " ";
        cout << endl;
    }

    ~CSRVertexGroup()
    {
        MemoryAPI::free_array(ids);
    }

    #ifdef __USE_GPU__
    void move_to_host()
    {
        if(size > 0)
            MemoryAPI::move_array_to_host(ids, size);
    }
    #endif

    #ifdef __USE_GPU__
    void move_to_device()
    {
        if(size > 0)
            MemoryAPI::move_array_to_device(ids, size);
    }
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
