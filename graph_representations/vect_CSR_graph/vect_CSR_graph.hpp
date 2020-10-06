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
void VectCSRGraph<_TVertexValue, _TEdgeWeight>::import_graph(EdgesListGraph<_TVertexValue, _TEdgeWeight> &_el_graph)
{
    this->vertices_count = _el_graph.get_vertices_count();
    this->edges_count = _el_graph.get_edges_count();

    double t1 = omp_get_wtime();
    outgoing_edges->import_and_preprocess(_el_graph);
    double t2 = omp_get_wtime();
    cout << "outgoing conversion time: " << t2 - t1 << " sec" << endl;

    _el_graph.transpose();

    t1 = omp_get_wtime();
    incoming_edges->import_and_preprocess(_el_graph);
    t2 = omp_get_wtime();
    cout << "incoming conversion time: " << t2 - t1 << " sec" << endl;

    _el_graph.transpose();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> *VectCSRGraph<_TVertexValue, _TEdgeWeight>::get_direction_graph_ptr(TraversalDirection _direction)
{
    if(_direction == SCATTER_TRAVERSAL)
    {
        return get_outgoing_graph_ptr();
    }
    else if(_direction == GATHER_TRAVERSAL)
    {
        return get_incoming_graph_ptr();
    }
    else
    {
        throw "Error in ExtendedCSRGraph::get_direction_graph_ptr, incorrect _direction type";
        return NULL;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void VectCSRGraph<_TVertexValue, _TEdgeWeight>::print()
{
    outgoing_edges->print();
    incoming_edges->print();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
template <typename _TVertexValue, typename _TEdgeWeight>
void VectCSRGraph<_TVertexValue, _TEdgeWeight>::print_with_weights(EdgesArrayNec<_TVertexValue, _TEdgeWeight, _TEdgeWeight> &_weights)
{
    outgoing_edges->print(_weights, SCATTER_TRAVERSAL);
    incoming_edges->print(_weights, GATHER_TRAVERSAL);
}
*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
