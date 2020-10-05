#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
template <typename ComputeOperation>
void GraphAbstractionsNEC<_TVertexValue, _TEdgeWeight>::compute_worker(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                       FrontierNEC<_TVertexValue, _TEdgeWeight> &_frontier,
                                                                       ComputeOperation &&compute_op)
{
    /*#ifdef __PRINT_API_PERFORMANCE_STATS__
    double t1 = omp_get_wtime();
    #pragma omp barrier
    #endif

    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    const long long int *vertex_pointers = vertex_pointers;

    int max_frontier_size = _frontier.max_size;

    if(_frontier.type == ALL_ACTIVE_FRONTIER)
    {
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma omp for schedule(static)
        for(int src_id = 0; src_id < max_frontier_size; src_id++)
        {
            int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
            int vector_index = get_vector_index(src_id);
            compute_op(src_id, connections_count, vector_index);
        }
    }
    else if((_frontier.type == DENSE_FRONTIER) || (_frontier.type == SPARSE_FRONTIER)) // TODO FIX SPARSE
    {
        int *frontier_flags = _frontier.flags;

        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        #pragma _NEC vector
        #pragma omp for schedule(static)
        for(int src_id = 0; src_id < max_frontier_size; src_id++)
        {
            if(frontier_flags[src_id] == IN_FRONTIER_FLAG)
            {
                int connections_count = vertex_pointers[src_id + 1] - vertex_pointers[src_id];
                int vector_index = get_vector_index(src_id);
                compute_op(src_id, connections_count, vector_index);
            }
        }
    }

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    #pragma omp barrier
    double t2 = omp_get_wtime();
    #pragma omp master
    {
        INNER_WALL_TIME += t2 - t1;
        INNER_COMPUTE_TIME += t2 - t1;

        double work = _frontier.size();
        cout << "compute time: " << (t2 - t1)*1000.0 << " ms" << endl;
        cout << "compute BW: " << sizeof(int)*(COMPUTE_INT_ELEMENTS)*work/((t2-t1)*1e9) << " GB/s" << endl << endl;
    }
    #endif*/
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
template <typename ComputeOperation>
void GraphAbstractionsNEC<_TVertexValue, _TEdgeWeight>::compute(VectCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                FrontierNEC<_TVertexValue, _TEdgeWeight> &_frontier,
                                                                ComputeOperation &&compute_op)
{
    ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> *current_direction_graph;

    // TODO is frontier direction correct?

    if(_traversal_direction == SCATTER_TRAVERSAL)
    {
        current_direction_graph = _graph.get_outgoing_graph_ptr();
    }
    else if(_traversal_direction == GATHER_TRAVERSAL)
    {
        current_direction_graph = _graph.get_incoming_graph_ptr();
    }

    if(omp_in_parallel())
    {
        #pragma omp barrier
        compute_worker(*current_direction_graph, _frontier, compute_op);
        #pragma omp barrier
    }
    else
    {
        #pragma omp parallel
        {
            compute_worker(*current_direction_graph, _frontier, compute_op);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
