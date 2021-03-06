#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
EdgesArray_Sharded<_T>::EdgesArray_Sharded(ShardedCSRGraph &_graph)
{
    long long edges_count = _graph.get_edges_count();

    long long edges_check = 0;
    this->total_array_size = 0;
    shards_number = _graph.get_shards_number();
    for(int sh = 0; sh < shards_number; sh++)
    {
        edges_check += _graph.get_edges_count_outgoing_shard(sh);
        this->total_array_size += _graph.get_edges_count_outgoing_shard(sh);
        this->total_array_size += _graph.get_edges_count_incoming_shard(sh);
        this->total_array_size += _graph.get_edges_count_in_ve_outgoing_shard(sh);
        this->total_array_size += _graph.get_edges_count_in_ve_incoming_shard(sh);
    }
    MemoryAPI::allocate_array(&this->edges_data, this->total_array_size);

    // set ptrs phase
    outgoing_csr_shards_ptrs.clear();
    incoming_csr_shards_ptrs.clear();
    outgoing_ve_shards_ptrs.clear();
    incoming_ve_shards_ptrs.clear();

    outgoing_csr_shards_sizes.clear();
    incoming_csr_shards_sizes.clear();
    outgoing_ve_shards_sizes.clear();
    incoming_ve_shards_sizes.clear();

    // set pointers and sizes for each shard in the OUTGOING part
    long long local_ptr = 0;
    for(int sh = 0; sh < shards_number; sh++)
    {
        long long csr_size = _graph.get_edges_count_outgoing_shard(sh);
        long long ve_size = _graph.get_edges_count_in_ve_outgoing_shard(sh);

        outgoing_csr_shards_sizes.push_back(csr_size);
        outgoing_ve_shards_sizes.push_back(ve_size);

        outgoing_csr_shards_ptrs.push_back(&this->edges_data[local_ptr]);
        local_ptr += _graph.get_edges_count_outgoing_shard(sh);
        outgoing_ve_shards_ptrs.push_back(&this->edges_data[local_ptr]);
        local_ptr += _graph.get_edges_count_in_ve_outgoing_shard(sh);
    }

    // set pointers and sizes for each shard in the INCOMING part (if required)
    for(int sh = 0; sh < shards_number; sh++)
    {
        long long csr_size = _graph.get_edges_count_incoming_shard(sh);
        long long ve_size = _graph.get_edges_count_in_ve_incoming_shard(sh);

        incoming_csr_shards_sizes.push_back(csr_size);
        incoming_ve_shards_sizes.push_back(ve_size);

        incoming_csr_shards_ptrs.push_back(&this->edges_data[local_ptr]);
        local_ptr += _graph.get_edges_count_incoming_shard(sh);
        incoming_ve_shards_ptrs.push_back(&this->edges_data[local_ptr]);
        local_ptr += _graph.get_edges_count_in_ve_incoming_shard(sh);
    }

    this->graph_ptr = &_graph;
    this->is_copy = false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
EdgesArray_Sharded<_T>::EdgesArray_Sharded(const EdgesArray_Sharded<_T> &_copy_obj)
{
    this->graph_ptr = _copy_obj.graph_ptr;
    this->edges_data = _copy_obj.edges_data;

    shards_number = _copy_obj.shards_number;
    outgoing_csr_shards_ptrs = _copy_obj.outgoing_csr_shards_ptrs;
    outgoing_ve_shards_ptrs = _copy_obj.outgoing_ve_shards_ptrs;
    incoming_csr_shards_ptrs = _copy_obj.incoming_csr_shards_ptrs;
    incoming_ve_shards_ptrs = _copy_obj.incoming_ve_shards_ptrs;

    outgoing_csr_shards_sizes = _copy_obj.outgoing_csr_shards_sizes;
    outgoing_ve_shards_sizes = _copy_obj.outgoing_ve_shards_sizes;
    incoming_csr_shards_sizes = _copy_obj.incoming_csr_shards_sizes;
    incoming_ve_shards_sizes = _copy_obj.incoming_ve_shards_sizes;

    this->total_array_size = _copy_obj.total_array_size;
    this->is_copy = true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
EdgesArray_Sharded<_T>::~EdgesArray_Sharded()
{
    if(!this->is_copy)
    {
        MemoryAPI::free_array(this->edges_data);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_Sharded<_T>::set_all_constant(_T _const)
{
    MemoryAPI::set(this->edges_data, _const, this->total_array_size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_Sharded<_T>::set_all_random(_T _max_rand)
{
    for(int sh = 0; sh < shards_number; sh++)
    {
        set_shard_all_random(sh, _max_rand);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_Sharded<_T>::set_shard_all_random(int _shard_id, _T _max_rand)
{
    // TODO using EdgesArray
    throw "EdgesArray_Sharded<_T>::set_shard_all_random not implemented yet";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_Sharded<_T>::print()
{
    cout << "Edges Array (Sharded CSR)" << endl;
    for(int sh = 0; sh < shards_number; sh++)
    {
        cout << "shard № " << sh << ": " << endl;
        print_shard(sh);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_Sharded<_T>::print_shard(int _shard_id)
{
    cout << "outgoing csr: ";
    for(long long i = 0; i < outgoing_csr_shards_sizes[_shard_id]; i++)
    {
        cout << outgoing_csr_shards_ptrs[_shard_id][i] << " ";
    }
    cout << endl;

    cout << "outgoing ve: ";
    for(long long i = 0; i < outgoing_ve_shards_sizes[_shard_id]; i++)
    {
        cout << outgoing_ve_shards_ptrs[_shard_id][i] << " ";
    }
    cout << endl;

    cout << "incoming csr: ";
    for(long long i = 0; i < incoming_csr_shards_sizes[_shard_id]; i++)
    {
        cout << incoming_csr_shards_ptrs[_shard_id][i] << " ";
    }
    cout << endl;

    cout << "outgoing ve: ";
    for(long long i = 0; i < incoming_ve_shards_sizes[_shard_id]; i++)
    {
        cout << incoming_ve_shards_ptrs[_shard_id][i] << " ";
    }
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_Sharded<_T>::copy_el_weights(int _shard_id, const EdgesArray_EL<_T> &_el_data)
{
    ShardedCSRGraph *sharded_graph_ptr = (ShardedCSRGraph *)this->graph_ptr;

    // allocated edges reorder buffer
    _T *buffer;
    long long shard_edges_count = sharded_graph_ptr->get_outgoing_shard_ptr(_shard_id)->get_edges_count();
    MemoryAPI::allocate_array(&buffer, shard_edges_count);

    // copy edges list weights to sharded weights
    sharded_graph_ptr->get_outgoing_shard_ptr(_shard_id)->reorder_and_copy_edges_from_original_to_sorted(outgoing_csr_shards_ptrs[_shard_id], _el_data.get_ptr());
    sharded_graph_ptr->get_outgoing_shard_ptr(_shard_id)->get_ve_ptr()->copy_array_from_csr_to_ve(outgoing_ve_shards_ptrs[_shard_id],
                                                                                                  outgoing_csr_shards_ptrs[_shard_id]);

    // copy edges list weights to sharded weights
    /*sharded_graph_ptr->get_incoming_shard_ptr(_shard_id)->reorder_and_copy_edges_from_original_to_sorted(incoming_csr_shards_ptrs[_shard_id], _el_data.get_ptr());
    sharded_graph_ptr->get_incoming_shard_ptr(_shard_id)->get_ve_ptr()->copy_array_from_csr_to_ve(incoming_ve_shards_ptrs[_shard_id],
                                                                                                  incoming_csr_shards_ptrs[_shard_id]);
    */
     // TODO check last part correctness

    MemoryAPI::free_array(buffer);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void EdgesArray_Sharded<_T>::operator = (const EdgesArray_EL<_T> &_el_data)
{
    for(int sh = 0; sh < shards_number; sh++)
    {
        copy_el_weights(sh, _el_data);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template class EdgesArray_Sharded<int>;
template class EdgesArray_Sharded<float>;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
