#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectCSRGraph::import_graph(EdgesListGraph &_el_graph)
{
    this->vertices_count = _el_graph.get_vertices_count();
    this->edges_count = _el_graph.get_edges_count();

    double t1 = omp_get_wtime();
    outgoing_graph->import_and_preprocess(_el_graph, NULL);
    double t2 = omp_get_wtime();
    cout << "outgoing conversion time: " << t2 - t1 << " sec" << endl;

    _el_graph.transpose();

    this->resize_helper_arrays();

    t1 = omp_get_wtime();
    incoming_graph->import_and_preprocess(_el_graph, edges_reorder_indexes);
    t2 = omp_get_wtime();
    cout << "incoming conversion time: " << t2 - t1 << " sec" << endl;

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
