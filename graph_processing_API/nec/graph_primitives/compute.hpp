#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template <typename _TVertexValue, typename _TEdgeWeight, typename ComputeOperation>
void GraphPrimitivesNEC::compute(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                 FrontierNEC &_frontier,
                                 ComputeOperation &&compute_op)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    const long long int *vertex_pointers = outgoing_ptrs;

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
                int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
                compute_op(src_id, connections_count);
            }
        }

        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        for(int src_id = max_frontier_size - VECTOR_LENGTH; src_id < max_frontier_size; src_id++)
        {
            int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
            compute_op(src_id, connections_count);
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
                {
                    int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
                    compute_op(src_id, connections_count);
                }
            }
        }

        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        for(int src_id = max_frontier_size - VECTOR_LENGTH; src_id < max_frontier_size; src_id++)
        {
            if(frontier_flags[src_id] == NEC_IN_FRONTIER_FLAG)
            {
                int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
                compute_op(src_id, connections_count);
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
