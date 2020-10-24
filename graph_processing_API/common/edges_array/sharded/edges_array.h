#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class EdgesArray_Sharded : public EdgesArray<_T>
{
private:
    vector<_T*> outgoing_csr_shards_ptrs;
    vector<_T*> outgoing_ve_shards_ptrs;
    vector<_T*> incoming_csr_shards_ptrs;
    vector<_T*> incoming_ve_shards_ptrs;

    vector<long long> outgoing_csr_shards_sizes;
    vector<long long> outgoing_ve_shards_sizes;
    vector<long long> incoming_csr_shards_sizes;
    vector<long long> incoming_ve_shards_sizes;

    int shards_number;

    void print_shard(int _shard_id);
    void set_shard_all_random(int _shard_id, _T _max_rand);

    void copy_el_weights(int _shard_id, const EdgesArray_EL<_T> &_el_data);
public:
    /* constructors and destructors */
    EdgesArray_Sharded(ShardedCSRGraph &_graph);
    EdgesArray_Sharded(const EdgesArray_Sharded<_T> &_copy_obj);
    ~EdgesArray_Sharded();

    /* initialization API */
    void set_all_constant(_T _const);
    void set_all_random(_T _max_rand);

    /* print API */
    void print();

    /* remaining API */
    void operator = (const EdgesArray_EL<_T> &_el_data);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "edges_array.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
