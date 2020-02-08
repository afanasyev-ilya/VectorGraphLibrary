#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierNEC::FrontierNEC(int _vertices_count)
{
    max_frontier_size = _vertices_count;
    frontier_flags = new int[max_frontier_size];

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < max_frontier_size; i++)
    {
        frontier_flags[i] = NEC_NOT_IN_FRONTIER_FLAG;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierNEC::~FrontierNEC()
{
    delete []frontier_flags;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Condition>
void FrontierNEC::filter(Condition condition_op)
{
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < max_frontier_size; i++)
    {
        if(condition_op(i))
            frontier_flags[i] = NEC_IN_FRONTIER_FLAG;
        else
            frontier_flags[i] = NEC_NOT_IN_FRONTIER_FLAG;
    }

    // if few vertices generate sparse
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int FrontierNEC::size()
{
    int size = 0;

    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp parallel for schedule(static) reduction(+: size)
    for(int i = 0; i < max_frontier_size; i++)
    {
        size += frontier_flags[i];
    }

    return size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
