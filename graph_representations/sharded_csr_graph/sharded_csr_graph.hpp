#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ShardedCSRGraph::ShardedCSRGraph(SupportedDirection _supported_direction)
{
    this->vertices_count = 1;
    this->edges_count = 1;
    this->graph_type = SHARDED_CSR_GRAPH;
    max_cached_vertices = 1;
    shards_number = 1;
    this->supported_direction = _supported_direction; // need to do this before init!

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

bool ShardedCSRGraph::save_to_binary_file(string _file_name)
{
    FILE * graph_file = fopen(_file_name.c_str(), "wb");
    if(graph_file == NULL)
        return false;

    if(!incoming_is_stored())
    {
        throw "Error in ShardedCSRGraph::save_to_binary_file : saved graph must have both directions";
    }

    int vertices_count = this->vertices_count;
    long long edges_count = this->edges_count;
    fwrite(reinterpret_cast<const char*>(&(this->graph_type)), sizeof(GraphType), 1, graph_file);
    fwrite(reinterpret_cast<const char*>(&this->vertices_count), sizeof(int), 1, graph_file);
    fwrite(reinterpret_cast<const char*>(&this->edges_count), sizeof(long long), 1, graph_file);
    fwrite(reinterpret_cast<const char*>(&this->shards_number), sizeof(int), 1, graph_file);

    for(int shard_id = 0; shard_id < shards_number; shard_id++)
    {
        outgoing_shards[shard_id].save_main_content_to_binary_file(graph_file);
    }

    for(int shard_id = 0; shard_id < shards_number; shard_id++)
    {
        incoming_shards[shard_id].save_main_content_to_binary_file(graph_file);
    }

    fclose(graph_file);
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool ShardedCSRGraph::load_from_binary_file(string _file_name)
{
    FILE * graph_file = fopen(_file_name.c_str(), "rb");
    if(graph_file == NULL)
        return false;

    fread(reinterpret_cast<char*>(&(this->graph_type)), sizeof(GraphType), 1, graph_file);
    if(this->graph_type != SHARDED_CSR_GRAPH)
    {
        throw "Error in ShardedCSRGraph::load_from_binary_file : graph type in file is not equal to SHARDED_CSR_GRAPH";
    }

    fread(reinterpret_cast<char*>(&this->vertices_count), sizeof(int), 1, graph_file);
    fread(reinterpret_cast<char*>(&this->edges_count), sizeof(long long), 1, graph_file);
    fread(reinterpret_cast<char*>(&this->shards_number), sizeof(int), 1, graph_file);

    resize(shards_number, this->vertices_count);

    for(int shard_id = 0; shard_id < shards_number; shard_id++)
    {
        if(outgoing_is_stored())
            outgoing_shards[shard_id].load_main_content_from_binary_file(graph_file);
        else  // TODO this should be equal to skip
            incoming_shards[shard_id].load_main_content_from_binary_file(graph_file);
    }

    for(int shard_id = 0; shard_id < shards_number; shard_id++)
    {
        if(incoming_is_stored())
            incoming_shards[shard_id].load_main_content_from_binary_file(graph_file);
        else  // TODO this should be equal to skip
            outgoing_shards[shard_id].load_main_content_from_binary_file(graph_file);
    }

    fclose(graph_file);
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
