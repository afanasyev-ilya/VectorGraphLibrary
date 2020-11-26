#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void VectCSRGraph::move_to_device()
{
    Timer tm;
    tm.start();
    if(this->graph_on_device)
        return;

    if(outgoing_is_stored())
        outgoing_graph->move_to_device();

    if(incoming_is_stored())
        incoming_graph->move_to_device();

    this->graph_on_device = true;

    tm.end();
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("VectCSRGraph::move_to_device", this->get_size());
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void VectCSRGraph::move_to_host()
{
    Timer tm;
    tm.start();
    if(!this->graph_on_device)
        return;

    if(outgoing_is_stored())
        outgoing_graph->move_to_host();

    if(incoming_is_stored())
        incoming_graph->move_to_host();

    this->graph_on_device = false;
    tm.end();

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("VectCSRGraph::move_to_host", this->get_size());
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
