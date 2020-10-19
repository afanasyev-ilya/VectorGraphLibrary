#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void VectCSRGraph::move_to_device()
{
    outgoing_graph->move_to_device();
    incoming_graph->move_to_device();

    MemoryAPI::move_array_to_device(edges_reorder_indexes, this->edges_count);
    MemoryAPI::move_array_to_device(edges_reorder_indexes, this->edges_count);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void VectCSRGraph::move_to_host()
{
    outgoing_graph->move_to_host();
    incoming_graph->move_to_host();

    MemoryAPI::move_array_to_host(edges_reorder_indexes, this->edges_count);
    MemoryAPI::move_array_to_host(edges_reorder_indexes, this->edges_count);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
