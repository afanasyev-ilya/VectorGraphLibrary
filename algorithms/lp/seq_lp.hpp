#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void LP::seq_lp(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph, int *_labels)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);

    // TODO seq algorithm

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    cout << "sequential check labels: " << endl;
    PerformanceStats::component_stats(_labels, vertices_count);
    #endif
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
