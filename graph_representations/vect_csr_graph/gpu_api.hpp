#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void VectCSRGraph::move_to_device()
{
    Timer tm;
    tm.start();
    if(this->graph_on_device)
        return;
    outgoing_graph->move_to_device();
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
    outgoing_graph->move_to_host();
    incoming_graph->move_to_host();

    this->graph_on_device = false;
    tm.end();

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_bandwidth_stats("VectCSRGraph::move_to_host", this->get_size());
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
