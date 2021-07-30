#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class UserDataContainer
{
public:
    virtual void reorder_from_original_to_shard(TraversalDirection _direction, int _shard_id) = 0;
    virtual void reorder_from_shard_to_original(TraversalDirection _direction, int _shard_id) = 0;

    virtual void reorder(TraversalDirection _output_dir) = 0;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum CachedMode
{
    USE_CACHED_MODE = 0,
    DONT_USE_CACHED_MODE = 1
};

#define CACHED_VERTICES 3500
#define CACHE_STEP 7

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class VerticesArray : public UserDataContainer
{
private:
    BaseGraph *graph_ptr;

    ObjectType object_type;
    TraversalDirection direction;

    _T *vertices_data;
    int vertices_count;

    _T *cached_data;
    CachedMode cached_mode;

    bool is_copy;

    void allocate_cached_array();
    void free_cached_array();
public:
    /* constructors and destructors */
    VerticesArray(VectCSRGraph &_graph, TraversalDirection _direction = SCATTER, CachedMode _cached_mode = DONT_USE_CACHED_MODE);
    VerticesArray(ShardedCSRGraph &_graph, TraversalDirection _direction = SCATTER, CachedMode _cached_mode = DONT_USE_CACHED_MODE);
    VerticesArray(EdgesListGraph &_graph, TraversalDirection _direction = ORIGINAL, CachedMode _cached_mode = DONT_USE_CACHED_MODE);

    VerticesArray(const VerticesArray<_T> &_copy_obj);
    ~VerticesArray();

    /* get/set API */
    _T *get_ptr() {return vertices_data;};
    ObjectType get_object_type() {return object_type;};
    int size() {return vertices_count;};

    #ifdef __USE_GPU__
    __host__ __device__ inline _T get(int _idx) const {return this->vertices_data[_idx];};
    __host__ __device__ inline _T set(int _idx, _T _val) const {this->vertices_data[_idx] = _val;};
    __host__ __device__ inline _T& operator[] (int _idx) const { return vertices_data[_idx]; };
    #else
    inline _T get(int _idx) const {return this->vertices_data[_idx];};
    inline void set(int _idx, _T _val) {this->vertices_data[_idx] = _val;};
    inline _T& operator[](int _idx) { return vertices_data[_idx]; }
    inline _T& operator[] (int _idx) const { return vertices_data[_idx]; };
    #endif

    inline _T cached_load(int _idx, _T *_private_data);

    inline void prefetch_data_into_cache();
    inline _T* get_private_data_pointer();

    /* direction API */
    TraversalDirection get_direction() {return direction;};
    void set_direction(TraversalDirection _direction) {direction = _direction;};

    /* initialization API */
    void set_all_constant(_T _const);
    void set_all_random(_T _max_rand);

    /* print API */
    void print();
    void print(string _name);

    /* GPU specific (copy) API */
    #ifdef __USE_GPU__
    void move_to_device();
    void move_to_host();
    #endif

    // allows to reorder vertices for a specific shard
    void reorder_from_original_to_shard(TraversalDirection _direction, int _shard_id);
    void reorder_from_shard_to_original(TraversalDirection _direction, int _shard_id);

    // allows to reorder verticesArray in arbitrary direction
    void reorder(TraversalDirection _output_dir);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "gpu_api.hpp"
#include "vertices_array.hpp"
#include "reorder.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
