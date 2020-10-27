#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ShardedCSRGraph::ShardedCSRGraph(SupportedDirection _supported_direction)
{
    this->vertices_count = 1;
    this->edges_count = 1;
    this->graph_type = SHARDED_CSR_GRAPH;
    max_cached_vertices = 1;
    shards_number = 1;
    init(shards_number, this->vertices_count);

    this->supported_direction = _supported_direction;
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
    if(can_use_scatter())
    {
        outgoing_shards = new UndirectedCSRGraph[_shards_count]; // MemoryAPI doesnt work here
    }
    if(can_use_gather())
    {
        incoming_shards = new UndirectedCSRGraph[_shards_count];
    }
    MemoryAPI::allocate_array(&vertices_reorder_buffer, _vertices_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ShardedCSRGraph::free()
{
    if(can_use_scatter())
    {
        delete[]outgoing_shards;
    }
    if(can_use_gather())
    {
        delete[]incoming_shards;
    }
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

    if(can_use_scatter())
    {
        for (int current_shard = 0; current_shard < shards_number; current_shard++)
        {
            long long csr_size = this->get_edges_count_outgoing_shard(current_shard);
            long long ve_size = this->get_edges_count_in_ve_outgoing_shard(current_shard);
            direction_shift += (csr_size + ve_size);
        }
    }
    return direction_shift;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

long long ShardedCSRGraph::get_shard_shift(int _shard_id, TraversalDirection _direction)
{
    long long shard_shift = 0;
    for (int current_shard = 0; current_shard < _shard_id; current_shard++)
    {
        long long csr_size = 0, ve_size = 0;
        if(_direction == SCATTER)
        {
            csr_size = this->get_edges_count_outgoing_shard(current_shard);
            ve_size = this->get_edges_count_in_ve_outgoing_shard(current_shard);
        }
        else if(_direction == GATHER)
        {
            csr_size = this->get_edges_count_incoming_shard(current_shard);
            ve_size = this->get_edges_count_in_ve_incoming_shard(current_shard);
        }
        else
        {
            throw "Error in ShardedCSRGraph::get_shard_shift : unsupported direction";
        }
        shard_shift += (csr_size + ve_size);
    }
    return shard_shift;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int ShardedCSRGraph::select_random_vertex(TraversalDirection _direction)
{
    int vertex_id = 0;
    if(_direction == SCATTER || _direction == ORIGINAL)
    {
        vertex_id = outgoing_shards[0].select_random_vertex();
    }
    else if(_direction == GATHER)
    {
        vertex_id = incoming_shards[0].select_random_vertex();
    }
    return vertex_id;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
