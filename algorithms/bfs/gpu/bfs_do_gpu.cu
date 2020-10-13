#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../../architectures.h"
#define INT_ELEMENTS_PER_EDGE 3.0
//#define __PRINT_API_PERFORMANCE_STATS__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../../graph_processing_API/gpu/cuda_API_include.h"
#include "../change_state/change_state.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

#define FIRST_LEVEL_VERTEX 1
#define UNVISITED_VERTEX -1

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define VERTICES_IN_VECTOR_EXTENSION 10

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ bottom_up_kernel(const long long *_vertex_pointers,
                                 const int *_adjacent_ids,
                                 const int _vertices_count,
                                 const int *_vector_extension,
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

            /*bool connected = false;
            #pragma unroll
            for(int i = 0; i < VERTICES_IN_VECTOR_EXTENSION; i++)
            {
                if(i < connections_count)
                {
                    int dst_id = _vector_extension[i * _vertices_count + src_id];
                    if (_levels[dst_id] == _current_level)
                    {
                        _levels[src_id] = _current_level + 1;
                        //atomicAdd(_vis, 1);
                        connected = true;
                        break;
                    }
                }
                else
                {
                    break;
                }
            }

            if(!connected)
            {
                for (int edge_pos = start + VERTICES_IN_VECTOR_EXTENSION; edge_pos < end; edge_pos++)
                {
                    int dst_id = _adjacent_ids[edge_pos];
                    if (_levels[dst_id] == _current_level)
                    {
                        _levels[src_id] = _current_level + 1;
                        atomicAdd(_vis, 1);
                        break;
                    }
                }
            }*/
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ ve_bottom_up_kernel(const long long *_vertex_pointers,
                                    const int *_adjacent_ids,
                                    const int _vertices_count,
                                    int *_vector_extension,
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
            for(int i = 0; i < VERTICES_IN_VECTOR_EXTENSION; i++)
            {
                int dst_id = _vector_extension[i * _vertices_count + src_id];
                if ((i < connections_count) && (_levels[dst_id] == _current_level))
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


void bottom_up_step(UndirectedCSRGraph &_graph,
                    FrontierGPU &_frontier,
                    GraphPrimitivesGPU &_graph_API,
                    const long long *_vertex_pointers,
                    const int *_adjacent_ids,
                    const int _vertices_count,
                    int *_vector_extension,
                    int *_levels,
                    int _current_level,
                    int *_vis,
                    bool _use_vector_extension)
{
    bottom_up_kernel<<< (_vertices_count - 1)/BLOCK_SIZE + 1, BLOCK_SIZE >>>(_vertex_pointers, _adjacent_ids, _vertices_count, _vector_extension, _levels, _current_level, _vis);
    cudaDeviceSynchronize();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ init_vector_extension_kernel(const long long *_vertex_pointers,
                                             const int *_adjacent_ids,
                                             const int _vertices_count,
                                             int *_vector_extension)
{
    const int src_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (src_id < _vertices_count)
    {
        long long start = _vertex_pointers[src_id];
        long long end = _vertex_pointers[src_id + 1];
        int connections_count = end - start;
        for (int i = 0; i < min(connections_count, VERTICES_IN_VECTOR_EXTENSION); i++)
        {
            _vector_extension[_vertices_count * i + src_id] = _adjacent_ids[start + i];
        }
    }
}

void init_vector_extension(const long long *_vertex_pointers,
                           const int *_adjacent_ids,
                           const int _vertices_count,
                           int *_vector_extension)
{
    init_vector_extension_kernel<<< (_vertices_count - 1)/BLOCK_SIZE + 1, BLOCK_SIZE >>>(_vertex_pointers, _adjacent_ids, _vertices_count, _vector_extension);
    cudaDeviceSynchronize();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void direction_optimizing_wrapper(UndirectedCSRGraph &_graph,
                                  int *_levels,
                                  int _source_vertex, int &_iterations_count)
{
    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);
    GraphPrimitivesGPU graph_API;
    FrontierGPU frontier(_graph.get_vertices_count());

    // unroll here - 4-5x
    int *vector_extension;
    /*MemoryAPI::allocate_device_array(&vector_extension, vertices_count * VERTICES_IN_VECTOR_EXTENSION);
    init_vector_extension(_graph.get_vertex_pointers(), _graph.get_adjacent_ids(), vertices_count, vector_extension);*/

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

    int current_frontier_size = 1, prev_frontier_size = 0;
    double t_begin = omp_get_wtime();
    do
    {
        double t1, t2;
        vis[0] = 0;

        auto reduce_op = [] __device__(int src_id, int position_in_frontier, int connections_count)->int
        {
            return connections_count;
        };
        in_lvl[0] = graph_API.reduce<int>(_graph, frontier, reduce_op, REDUCE_SUM);

        MemoryAPI::prefetch_managed_array(vis, 1);

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

            //graph_API.advance(_graph, frontier, edge_op, EMPTY_VERTEX_OP, EMPTY_VERTEX_OP, frontier, on_next_level);
            graph_API.advance(_graph, frontier, edge_op);
        }
        else if(current_state == BOTTOM_UP)
        {
            bool _use_vector_extension = false;
            //_use_vector_extension = true;
            bottom_up_step(_graph, frontier, graph_API, _graph.get_vertex_pointers(), _graph.get_adjacent_ids(), vertices_count, vector_extension, _levels, current_level, vis, _use_vector_extension);
        }

        if(vis[0] == 0)
        {
            break;
        }

        StateOfBFS next_state = gpu_change_state(current_frontier_size, vis[0], vertices_count, edges_count,
                current_state, vis[0], in_lvl[0], current_level, POWER_LAW_GRAPH);

        if(next_state == TOP_DOWN)
        {
            auto on_next_level = [_levels, current_level] __device__ (int src_id)->int
            {
                if(_levels[src_id] == (current_level + 1))
                    return IN_FRONTIER_FLAG;
                else
                    return NOT_IN_FRONTIER_FLAG;
            };
            graph_API.generate_new_frontier(_graph, frontier, on_next_level);
        }

        current_state = next_state;
        current_frontier_size = vis[0];
        prev_frontier_size = current_frontier_size;
        current_level++;

    } while(vis[0] > 0);
    double t_end = omp_get_wtime();
    //cout << "inner perf: " << edges_count / ((t_end - t_begin)*1e6) << " TEPS" << endl;

    _iterations_count = current_level;

    //MemoryAPI::free_device_array(vector_extension);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void direction_optimizing_wrapper<int, float>(UndirectedCSRGraph<int, float> &_graph, int *_levels,
                                                       int _source_vertex, int &_iterations_count);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

