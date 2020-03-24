#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template <typename _TVertexValue, typename _TEdgeWeight, typename ComputeOperation>
void GraphPrimitivesNEC::compute(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                 FrontierNEC &_frontier,
                                 ComputeOperation &&compute_op)
{
    int max_frontier_size = _frontier.max_size;

    if(_frontier.type == ALL_ACTIVE_FRONTIER)
    {
        #pragma omp parallel for schedule(static)
        for(int vec_start = 0; vec_start < max_frontier_size - VECTOR_LENGTH; vec_start += VECTOR_LENGTH)
        {
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = vec_start + i;
                compute_op(src_id);
            }
        }

        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        for(int src_id = max_frontier_size - VECTOR_LENGTH; src_id < max_frontier_size; src_id++)
        {
            compute_op(src_id);
        }
    }
    else if((_frontier.type == DENSE_FRONTIER) || (_frontier.type == SPARSE_FRONTIER)) // TODO FIX SPARSE
    {
        int *frontier_flags = _frontier.flags;

        #pragma omp parallel for schedule(static)
        for(int vec_start = 0; vec_start < max_frontier_size - VECTOR_LENGTH; vec_start += VECTOR_LENGTH)
        {
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = vec_start + i;
                if(frontier_flags[src_id] == NEC_IN_FRONTIER_FLAG)
                    compute_op(src_id);
            }
        }

        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        for(int src_id = max_frontier_size - VECTOR_LENGTH; src_id < max_frontier_size; src_id++)
        {
            if(frontier_flags[src_id] == NEC_IN_FRONTIER_FLAG)
                compute_op(src_id);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
