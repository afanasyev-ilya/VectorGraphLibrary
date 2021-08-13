#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct CSRVertexGroup
{
    int *ids;
    int size;
    int max_size;
    long long neighbours;

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
        MemoryAPI::copy(this->ids, _other_group.ids, this->size );
    }

    template <typename CopyCond>
    void copy_data_if(CSRVertexGroup &_full_group, CopyCond copy_cond)
    {
        cout << "copy if needed" << endl;
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
