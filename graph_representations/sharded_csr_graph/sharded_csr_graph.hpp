#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ShardedCSRGraph::ShardedCSRGraph()
{
    this->vertices_count = 1;
    this->edges_count = 1;
    this->graph_type = SHARDED_CSR_GRAPH;
    max_cached_vertices = 1;
    shards_number = 1;
    init(shards_number, this->vertices_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ShardedCSRGraph::resize(int _shards_count, int _vertices_count)
{
    free();
    init(_shards_count, _vertices_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ShardedCSRGraph::init(int _shards_count, int _vertices_count)
{
    outgoing_shards = new UndirectedCSRGraph[_shards_count]; // MemoryAPI doesnt work here
    incoming_shards = new UndirectedCSRGraph[_shards_count];
    MemoryAPI::allocate_array(&vertices_reorder_buffer, _vertices_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ShardedCSRGraph::free()
{
    delete []outgoing_shards;
    delete []incoming_shards;
    MemoryAPI::free_array(vertices_reorder_buffer);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ShardedCSRGraph::~ShardedCSRGraph()
{
    free();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

long long ShardedCSRGraph::get_direction_shift()
{
    long long direction_shift = 0;
    for (int current_shard = 0; current_shard < shards_number; current_shard++)
    {
        long long csr_size = this->get_edges_count_outgoing_shard(current_shard);
        long long ve_size = this->get_edges_count_in_ve_outgoing_shard(current_shard);
        direction_shift += (csr_size + ve_size);
    }
    return direction_shift;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

long long ShardedCSRGraph::get_shard_shift(int _shard_id)
{
    long long shard_shift = 0;
    for (int current_shard = 0; current_shard < _shard_id; current_shard++)
    {
        long long csr_size = this->get_edges_count_outgoing_shard(current_shard);
        long long ve_size = this->get_edges_count_in_ve_outgoing_shard(current_shard);
        shard_shift += (csr_size + ve_size);
    }
    return shard_shift;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
