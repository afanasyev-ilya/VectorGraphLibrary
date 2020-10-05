#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
class EdgesArrayNec
{
private:
    _T *outgoing_data;
    _T *incoming_data;
public:
    template <typename _TVertexValue, typename _TEdgeWeight>
    EdgesArrayNec(VectCSRGraph<_TVertexValue, _TEdgeWeight> &_graph);

    ~EdgesArrayNec();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

EdgesArrayNec::EdgesArrayNec(VectCSRGraph<_TVertexValue, _TEdgeWeight> &_graph)
{
    long long edges_in_csr = _graph.get_edges_count();
    long long edges_in_ve = 10;
    // программа может на разных итерациях использовать VE и нет, в зависимости от разреженности - как тут быть?
    // нужно хранить
    // при этом для двух направлений - разные VE (разного размера), вообще - задница. - редкий случай, можно ли забить?

    //
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
