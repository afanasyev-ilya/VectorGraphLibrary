#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
VectCSRGraph<_TVertexValue, _TEdgeWeight>::VectCSRGraph(int _vertices_count, long long _edges_count)
{
    this->vertices_count = _vertices_count;
    this->edges_count = _edges_count;
    outgoing_edges = new ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>(_vertices_count, _edges_count);
    incoming_edges = new ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>(_vertices_count, _edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
VectCSRGraph<_TVertexValue, _TEdgeWeight>::~VectCSRGraph()
{
    delete outgoing_edges;
    delete incoming_edges;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void VectCSRGraph<_TVertexValue, _TEdgeWeight>::import_graph(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_copy_graph)
{
    ASL_CALL(asl_library_initialize());
    asl_sort_t hnd;
    ASL_CALL(asl_sort_create_i32(&hnd, ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO));


    int kyi[5] = {1, 20, 4, 3, 2};
    asl_int_t vli[5] = {0, 1, 2, 3, 4};

    int kyo[5] = {0, 0, 0, 0, 0};
    asl_int_t vlo[5] = {0, 0, 0, 0, 0};

    //ASL_CALL(asl_sort_execute_i32(hnd, 5, kyi, vli, kyo, vlo));
    ASL_CALL(asl_sort_execute_i32(hnd, 5, kyi, vli, kyo, vli));

    for(int i = 0; i < 5; i ++)
        cout << kyo[i] << " - " << vli[i] << " for sorting " << kyi[i] << " -> " << kyi[vli[i]] <<  endl;

    ASL_CALL(asl_sort_destroy(hnd));
    ASL_CALL(asl_library_finalize());

    double t1 = omp_get_wtime();
    outgoing_edges->new_import_graph(_copy_graph);
    double t2 = omp_get_wtime();
    cout << "outgoing conversion time: " << t2 - t1 << " sec" << endl;

    /*t1 = omp_get_wtime();
    incoming_edges->new_import_graph(_copy_graph);
    t2 = omp_get_wtime();
    cout << "incoming conversion time: " << t2 - t1 << " sec" << endl;*/


}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
