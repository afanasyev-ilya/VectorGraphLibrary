#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierNEC::FrontierNEC(int _vertices_count)
{
    max_frontier_size = _vertices_count;
    current_frontier_size = 0;
    frontier_ids = new int[max_frontier_size];
    frontier_flags = new int[max_frontier_size];

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < max_frontier_size; i++)
    {
        frontier_ids[i] = 0;
        frontier_flags[i] = NEC_NOT_IN_FRONTIER_FLAG;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FrontierNEC::~FrontierNEC()
{
    delete []frontier_ids;
    delete []frontier_flags;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierNEC::set_all_active()
{
    current_frontier_size = max_frontier_size;

    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < max_frontier_size; i++)
    {
        frontier_ids[i] = i;
        frontier_flags[i] = NEC_IN_FRONTIER_FLAG;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
