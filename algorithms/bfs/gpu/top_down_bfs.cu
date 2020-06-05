#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 3.0

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../../graph_processing_API/gpu/cuda_API_include.h"
#include "../change_state/change_state.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

#define FIRST_LEVEL_VERTEX 1
#define UNVISITED_VERTEX -1

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void top_down_wrapper(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                      int *_levels,
                      int _source_vertex,
                      int &_iterations_count)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    GraphPrimitivesGPU graph_API;
    FrontierGPU frontier(_graph.get_vertices_count());

    frontier.set_all_active();

    auto init_levels = [_levels, _source_vertex] __device__ (int src_id, int position_in_frontier, int connections_count)
    {
        if(src_id == _source_vertex)
            _levels[_source_vertex] = FIRST_LEVEL_VERTEX;
        else
            _levels[src_id] = UNVISITED_VERTEX;
    };
    graph_API.compute(_graph, frontier, init_levels);

    auto on_first_level = [_levels] __device__ (int src_id)->int
    {
        if(_levels[src_id] == FIRST_LEVEL_VERTEX)
            return IN_FRONTIER_FLAG;
        else
            return NOT_IN_FRONTIER_FLAG;
    };
    graph_API.generate_new_frontier(_graph, frontier, on_first_level);

    int current_level = FIRST_LEVEL_VERTEX;
    while(frontier.size() > 0)
    {
        auto edge_op = [_levels, current_level] __device__ (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int frontier_pos)
        {
            int src_level = _levels[src_id];
            int dst_level = _levels[dst_id];
            if((src_level == current_level) && (dst_level == UNVISITED_VERTEX))
            {
                _levels[dst_id] = current_level + 1;
            }
        };

        graph_API.advance(_graph, frontier, edge_op);

        auto on_next_level = [_levels, current_level] __device__ (int src_id)->int
        {
            if(_levels[src_id] == (current_level + 1))
                return IN_FRONTIER_FLAG;
            else
                return NOT_IN_FRONTIER_FLAG;
        };

        graph_API.generate_new_frontier(_graph, frontier, on_next_level);

        current_level++;
    }
    _iterations_count = current_level;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ bottom_up_kernel(const long long *_vertex_pointers,
                                 const int *_adjacent_ids,
                                 const int _vertices_count,
                                 int *_levels,
                                 int _current_level,
                                 int *_vis)
{
    const int src_id = blockIdx.x * blockDim.x + threadIdx.x;

    if(src_id < _vertices_count)
    {
        if(_levels[src_id] == UNVISITED_VERTEX)
        {
            long long start = _vertex_pointers[src_id];
            long long end = _vertex_pointers[src_id + 1];
            int connections_count = end - start;
            for (int edge_pos = start; edge_pos < end; edge_pos++)
            {
                int dst_id = _adjacent_ids[edge_pos];
                if (_levels[dst_id] == _current_level)
                {
                    _levels[src_id] = _current_level + 1;
                    atomicAdd(_vis, 1);
                    break;
                }
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void bottom_up_step(const long long *_vertex_pointers,
                    const int *_adjacent_ids,
                    const int _vertices_count,
                    int *_levels,
                    int _current_level,
                    int *_vis)
{
    bottom_up_kernel<<< (_vertices_count - 1)/BLOCK_SIZE + 1, BLOCK_SIZE >>>(_vertex_pointers, _adjacent_ids, _vertices_count, _levels, _current_level, _vis);
    cudaDeviceSynchronize();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void direction_optimizing_wrapper(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                  int *_levels,
                                  int _source_vertex, int &_iterations_count)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    GraphPrimitivesGPU graph_API;
    FrontierGPU frontier(_graph.get_vertices_count());

    int *next_frontier_size;
    MemoryAPI::allocate_managed_array(&next_frontier_size, 1);

    auto init_levels = [_levels, _source_vertex] __device__ (int src_id, int position_in_frontier, int connections_count)
    {
        if(src_id == _source_vertex)
            _levels[_source_vertex] = FIRST_LEVEL_VERTEX;
        else
            _levels[src_id] = UNVISITED_VERTEX;
    };
    frontier.set_all_active();
    graph_API.compute(_graph, frontier, init_levels);

    frontier.clear();
    frontier.add_vertex(_graph, _source_vertex);

    int current_level = FIRST_LEVEL_VERTEX;
    StateOfBFS current_state = TOP_DOWN;

    int *vis;
    int *in_lvl;
    MemoryAPI::allocate_managed_array(&vis, 1);
    MemoryAPI::allocate_managed_array(&in_lvl, 1);
    vis[0] = 1;
    in_lvl[0] = 0;

    double total_time = 0;
    double reduce_time = 0, advance_time = 0, gnf_time = 0;

    int current_frontier_size = 1, prev_frontier_size = 0;
    double t_begin = omp_get_wtime();
    do
    {
        double t1, t2;
        cout << "state: " << current_state << endl;
        cout << "prev_frontier_size: " << prev_frontier_size << endl;
        vis[0] = 0;

        t1 = omp_get_wtime();
        auto reduce_op = [] __device__ (int src_id, int connections_count)->int
        {
            return connections_count;
        };
        in_lvl[0] = graph_API.reduce<int>(_graph, frontier, reduce_op, REDUCE_SUM);
        t2 = omp_get_wtime();
        reduce_time += t2 - t1;
        cout << "reduces time: " << 1000.0*(t2 - t1) << " ms" << endl;
        cout << "in lvl (for estimating next front): " << in_lvl[0] << " " << 100.0*in_lvl[0]/vertices_count << endl;

        MemoryAPI::prefetch_managed_array(vis, 1);

        t1 = omp_get_wtime();
        if(current_state == TOP_DOWN)
        {
            auto edge_op = [_levels, current_level, vis, in_lvl] __device__ (int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int frontier_pos)
            {
                if(_levels[dst_id] == UNVISITED_VERTEX)
                {
                    _levels[dst_id] = current_level + 1;
                    atomicAdd(vis, 1);
                }
            };

            auto EMPTY_VERTEX_OP = [] __device__(int src_id, int position_in_frontier, int connections_count){};

            auto on_next_level = [_levels, current_level] __device__ (int src_id)->int
            {
                if(_levels[src_id] == (current_level + 1))
                    return IN_FRONTIER_FLAG;
                else
                    return NOT_IN_FRONTIER_FLAG;
            };

            graph_API.advance(_graph, frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP, frontier, on_next_level);
        }
        else if(current_state == BOTTOM_UP)
        {
            bottom_up_step(_graph.get_outgoing_ptrs(), _graph.get_outgoing_ids(), vertices_count, _levels, current_level, vis);
        }
        t2 = omp_get_wtime();
        cout << "td/bu time: " << 1000.0*(t2 - t1) << " ms" << endl;
        total_time += t2 - t1;
        advance_time += t2 - t1;

        if(vis[0] == 0)
        {
            break;
        }

        StateOfBFS next_state = gpu_change_state(current_frontier_size, vis[0], vertices_count, edges_count,
                current_state, vis[0], in_lvl[0], current_level, POWER_LAW_GRAPH);

        if((current_state == BOTTOM_UP) && (next_state == TOP_DOWN))
        {
            t1 = omp_get_wtime();
            auto on_next_level = [_levels, current_level] __device__ (int src_id)->int
            {
                if(_levels[src_id] == (current_level + 1))
                    return IN_FRONTIER_FLAG;
                else
                    return NOT_IN_FRONTIER_FLAG;
            };
            graph_API.generate_new_frontier(_graph, frontier, on_next_level);
            t2 = omp_get_wtime();
            cout << "gnf time: " << 1000.0*(t2 - t1) << " ms" << endl;
            gnf_time += t2 - t1;
        }

        current_state = next_state;
        current_frontier_size = vis[0];
        prev_frontier_size = current_frontier_size;
        current_level++;

        cout << "vis: " << vis[0] << endl;
        cout << endl;
    } while(vis[0] > 0);
    double t_end = omp_get_wtime();
    cout << "inner time: " << total_time*1000 << " ms" <<  endl;
    cout << "inner 2 time: " << 1000*(t_end - t_begin) << " ms" << endl;
    cout << "reduce_time: " << reduce_time * 1000  << " ms" << endl;
    cout << "advance_time: " << advance_time * 1000  << " ms" << endl;
    cout << "gnf_time: " << gnf_time * 1000  << " ms" << endl;

    _iterations_count = current_level;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template void top_down_wrapper<int, float>(ExtendedCSRGraph<int, float> &_graph, int *_levels,
                                           int _source_vertex, int &_iterations_count);
template void direction_optimizing_wrapper<int, float>(ExtendedCSRGraph<int, float> &_graph, int *_levels,
                                                       int _source_vertex, int &_iterations_count);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

