#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// implements HITS algorithm
class HITS
{
public:

    template <typename _T>
    static double vgl_hits(VGL_Graph &_graph, VerticesArray<_T> &_auth, VerticesArray<_T> &_hub, int _num_steps);

    template <typename _T>
    static double seq_hits(VGL_Graph &_graph, VerticesArray<_T> &_auth, VerticesArray<_T> &_hub, int _num_steps);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "hits.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
