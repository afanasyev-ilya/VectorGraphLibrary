#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct CSRVertexGroup
{
    int *ids;
    int size;
    long long neighbours;

    CSRVertexGroup()
    {
        size = 1;
        neighbours = 0;
        MemoryAPI::allocate_array(&ids, size);
    }

    void resize(int _new_size)
    {
        size = _new_size;
        MemoryAPI::free_array(ids);
        MemoryAPI::allocate_array(&ids, _new_size + 1);
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
