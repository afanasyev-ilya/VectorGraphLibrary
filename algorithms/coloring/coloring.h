#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <bitset>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Coloring
{
public:
    template <typename _T>
    static void vgl_coloring(VGL_Graph &_graph, VerticesArray<_T> &_colors);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "coloring.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
