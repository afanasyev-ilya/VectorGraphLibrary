#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectCSRGraph::import_graph(EdgesListGraph &_el_graph)
{
    this->vertices_count = _el_graph.get_vertices_count();
    this->edges_count = _el_graph.get_edges_count();

    Timer tm;
    tm.start();
    outgoing_graph->import_and_preprocess(_el_graph, NULL);
    tm.end();
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_stats("VectCSR outgoing conversion");
    #endif

    _el_graph.transpose();

    this->resize_helper_arrays();

    tm.start();
    incoming_graph->import_and_preprocess(_el_graph, edges_reorder_indexes);
    tm.end();
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_stats("VectCSR incoming conversion");
    #endif

    _el_graph.transpose();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectCSRGraph::resize_helper_arrays()
{
    if(edges_reorder_indexes != NULL)
        MemoryAPI::free_array(edges_reorder_indexes);
    MemoryAPI::allocate_array(&edges_reorder_indexes, this->edges_count);
    MemoryAPI::set(edges_reorder_indexes, (long long)0, this->edges_count);

    if(vertices_reorder_buffer != NULL)
        MemoryAPI::free_array(vertices_reorder_buffer);
    MemoryAPI::allocate_array(&vertices_reorder_buffer, this->vertices_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
