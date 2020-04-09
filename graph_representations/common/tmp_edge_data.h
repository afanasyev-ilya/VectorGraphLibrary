#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <limits>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
struct TempEdgeData
{
    int dst_id;
    _TEdgeWeight weight;
    
    TempEdgeData(int _dst_id, _TEdgeWeight _weight)
    {
        dst_id = _dst_id;
        weight = _weight;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
bool edge_less(TempEdgeData<_TEdgeWeight> lhs, TempEdgeData<_TEdgeWeight> rhs)
{
    if(lhs.dst_id < rhs.dst_id)
        return true;
    if(lhs.dst_id == rhs.dst_id)
        return lhs.weight < rhs.weight;
    return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
bool edge_equal(TempEdgeData<_TEdgeWeight> lhs, TempEdgeData<_TEdgeWeight> rhs)
{
    if((lhs.dst_id == rhs.dst_id)/* && (std::abs(lhs.weight - rhs.weight) <
                            std::numeric_limits<_TEdgeWeight>::epsilon()*std::abs(lhs.weight + rhs.weight))*/)
        return true;
    return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
