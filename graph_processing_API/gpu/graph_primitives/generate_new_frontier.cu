#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct is_active
{
    __device__
    bool operator()(const int x)
    {
        return x != -1;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct is_not_active
{
    __device__
    bool operator()(const int x)
    {
        return x == -1;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Condition>
void __global__ copy_frontier_ids_kernel(int *_frontier_ids,
                                         int *_frontier_flags,
                                         const int _vertices_count,
                                         Condition cond)
{
    register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _vertices_count)
    {
        if(cond(idx) == true)
        {
            _frontier_ids[idx] = idx;
            _frontier_flags[idx] = IN_FRONTIER_FLAG;
        }
        else
        {
            _frontier_ids[idx] = -1;
            _frontier_flags[idx] = NOT_IN_FRONTIER_FLAG;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight, typename Condition>
void GraphPrimitivesGPU::generate_new_frontier(UndirectedCSRGraph &_graph,
                                               FrontierGPU &_frontier,
                                               Condition &&cond)
{
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    cudaDeviceSynchronize();
    double t1 = omp_get_wtime();
    #endif

    int vertices_count = _graph.get_vertices_count();
    _frontier.type = SPARSE_FRONTIER;

    SAFE_KERNEL_CALL((copy_frontier_ids_kernel<<<(vertices_count-1)/BLOCK_SIZE+1, BLOCK_SIZE>>>(_frontier.ids, _frontier.flags,
                                                                                                vertices_count, cond)));

    double t3 = omp_get_wtime();
    int *new_end = thrust::remove_if(thrust::device, _frontier.ids, _frontier.ids + vertices_count, is_not_active());
    _frontier.current_size = new_end - _frontier.ids;

    #ifdef __PRINT_API_PERFORMANCE_STATS__
    cudaDeviceSynchronize();
    double t2 = omp_get_wtime();
    INNER_WALL_TIME += t2 - t1;
    INNER_GNF_TIME += t2 - t1;
    double work = vertices_count;
    double kernel_time = t3 - t1;
    double thrust_time = t2 - t3;
    cout << "generated size: " << _frontier.size() << " vs " << _graph.get_vertices_count() << ", " << 100.0*_frontier.size()/vertices_count << "%" << endl;
    cout << "GNF time: " << (t2 - t1)*1000.0 << " ms" << endl;
    cout << "kernel BW: " << sizeof(int)*GNF_INT_ELEMENTS*work/(kernel_time*1e9) << " GB/s" << endl;
    cout << "thrust BW: " << sizeof(int)*2.0*work/(thrust_time*1e9) << " GB/s" << endl;
    cout << "GNF BW: " << sizeof(int)*(GNF_INT_ELEMENTS + 2.0)*work/((t2-t1)*1e9) << " GB/s" << endl << endl;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////