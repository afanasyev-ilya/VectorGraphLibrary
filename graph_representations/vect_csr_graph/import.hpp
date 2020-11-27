#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VectCSRGraph::import(EdgesListGraph &_el_graph)
{
    this->resize(_el_graph.get_vertices_count(), _el_graph.get_edges_count());

    Timer tm;
    if(outgoing_is_stored())
    {
        tm.start();
        outgoing_graph->import(_el_graph);
        tm.end();
        #ifdef __PRINT_API_PERFORMANCE_STATS__
        tm.print_time_stats("VectCSR outgoing conversion");
        #endif
    }

    _el_graph.transpose();

    if(incoming_is_stored())
    {
        tm.start();
        incoming_graph->import(_el_graph);
        tm.end();
        #ifdef __PRINT_API_PERFORMANCE_STATS__
        tm.print_time_stats("VectCSR incoming conversion");
        #endif
    }

    _el_graph.transpose();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
