#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void UndirectedCSRGraph::move_to_device()
{
    if(this->graph_on_device)
    {
        return;
    }
    
    this->graph_on_device = true;

    MemoryAPI::move_array_to_device(vertex_pointers, this->vertices_count + 1);
    MemoryAPI::move_array_to_device(adjacent_ids, this->edges_count);

    MemoryAPI::move_array_to_device(forward_conversion, this->vertices_count);
    MemoryAPI::move_array_to_device(backward_conversion, this->vertices_count);

    MemoryAPI::move_array_to_device(edges_reorder_indexes, this->edges_count);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void UndirectedCSRGraph::move_to_host()
{
    if(!this->graph_on_device)
    {
        return;
    }
    
    this->graph_on_device = false;

    MemoryAPI::move_array_to_host(vertex_pointers, this->vertices_count + 1);
    MemoryAPI::move_array_to_host(adjacent_ids, this->edges_count);

    MemoryAPI::move_array_to_host(forward_conversion, this->vertices_count);
    MemoryAPI::move_array_to_host(backward_conversion, this->vertices_count);

    MemoryAPI::move_array_to_host(edges_reorder_indexes, this->edges_count);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

