#pragma once
#define REDUCE_INITIAL -1
#define DECISION_BOUND 0.2

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../../architectures.h"
#define INT_ELEMENTS_PER_EDGE 3.0

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define LP_BOUNDARY_ACTIVE 1
#define LP_BOUNDARY_PASSIVE 2
#define LP_INNER 3

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <random>
#include <curand.h>
#include <curand_kernel.h>
#include "../lp_constants.h"
#include "../../../graph_processing_API/gpu/cuda_API_include.h"
#include "../../../external_libraries/moderngpu/src/moderngpu/kernel_segsort.hxx"
#include "../../../external_libraries/moderngpu/src/moderngpu/memory.hxx"
#include "../../../external_libraries/moderngpu/src/moderngpu/kernel_segreduce.hxx"
#include "../../../external_libraries/moderngpu/src/moderngpu/kernel_scan.hxx"
#include "active_conditions.cuh"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Puts a 1 in differences[i] if a label of next vertice is different from its label. 0 for cases with the same labels
__global__ void label_differences_advanced(int *differences, int *dest_labels, int edges_count)
{
    long long int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ((differences[i] != 1) && (i < edges_count))
    {
        if (dest_labels[i] != dest_labels[i + 1])
        {
            differences[i] = 1;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Get rid of repeated labels by setting indices to reduce_scan array
// i-th element of reduced_scan contains index of last entry of i element in scanned array
__global__ void count_labels(int *scanned_array, long long int edges_count, int *reduced_scan)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ((i < edges_count) && (scanned_array[i + 1] != scanned_array[i]))
    {
        reduced_scan[scanned_array[i]] = i;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Getting frequency of each label in reduced_scan
__global__ void frequency_count(int *frequencies, int *reduced_scan, long long int reduced_size)
{
    long long int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < reduced_size)
    {
        if ((i > 0))
        {
            frequencies[i] = reduced_scan[i] - reduced_scan[i - 1];
        }
        else
        {
            frequencies[0] = reduced_scan[0] + 1;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Update labels
__global__ void get_labels(int *reduce_result, int *reduced_scan, int *gathered_labels, int *_labels, int vertices_count, int *_updated)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < vertices_count)
    {
        if((reduce_result[i] != -1) && (_labels[i] != gathered_labels[reduced_scan[reduce_result[i]]]))
        {
            _labels[i] = gathered_labels[reduced_scan[reduce_result[i]]];
            _updated[0] = 1;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Service inidices to iterate over frequencies_array
__global__ void fill_indices(int *seg_reduce_indices, long long edges_count)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < edges_count)
    {
        seg_reduce_indices[i] = i;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void print_active_percentages(int *_node_states, int _vertices_count)
{
    int boundary_active_count = 0;
    int boundary_passive_count = 0;
    int inner_count = 0;
    for(int i = 0; i < _vertices_count; i++)
    {
        if(_node_states[i] == LP_INNER)
            inner_count++;
        if(_node_states[i] == LP_BOUNDARY_ACTIVE)
            boundary_active_count++;
        if(_node_states[i] == LP_BOUNDARY_PASSIVE)
            boundary_passive_count++;
    }

    cout << 100.0 * boundary_active_count / _vertices_count << " % boundary active" << endl;
    cout << 100.0 * boundary_passive_count / _vertices_count << " % boundary passive" << endl;
    cout << 100.0 * inner_count / _vertices_count << " % boundary inner" << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void print_active_ids(int *_node_states, int *_shifts, long long *_vertex_pointers, int _vertices_count)
{
    if(_vertices_count < 30)
    {
        cout << "ids of active vertices: " << endl;
        for (int src_id = 0; src_id < _vertices_count; src_id++)
        {
            if (_node_states[src_id] == LP_BOUNDARY_ACTIVE)
                printf("id=%d(CC=%ld, shift=%d) ", src_id, _vertex_pointers[src_id + 1] - _vertex_pointers[src_id], _shifts[src_id]);
        }
        cout << endl;
    }
}

template <typename _T>
void print_data(string _name, _T *_data, int _size)
{
    cout << _name << ": ";
    for(int i = 0; i < _size; i++)
    {
        cout << _data[i] << " ";
    }
    cout << endl << endl;
}

template <typename DataType, typename SegmentType>
void print_segmented_array(string _name, DataType *_data, SegmentType *_segments, int _segment_count, int _data_size)
{
    cout << _name << ": ";
    for(int i = 0; i < _data_size; i++)
    {
        cout << _data[i] << " ";
    }
    cout << endl;

    cout << _name << " with segments: ";

    for(int segment = 0; segment < _segment_count; segment++)
    {
        int start = _segments[segment];
        int end = _segments[segment + 1];
        cout << " (" << segment << ")" << "[";
        for(int i = start; i < end; i++)
        {
            if(i != (end - 1))
                cout << _data[i] << " ";
            else
                cout << _data[i];
        }
        cout << "] ";
    }
    cout << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename _TVertexValue, typename _TEdgeWeight>
void gpu_lp_wrapper(UndirectedCSRGraph &_graph,
                    int *_labels,
                    int &_iterations_count,
                    GpuActiveConditionType _gpu_active_condition_type,
                    int _max_iterations)
{
    LOAD_UNDIRECTED_CSR_GRAPH_DATA(_graph);
    GraphPrimitivesGPU graph_API;
    FrontierGPU frontier(_graph.get_vertices_count());

    mgpu::standard_context_t context;

    int *gathered_labels;
    int *tmp_work_buffer_for_seg_sort;
    int *new_ptr;
    int *label_differences;
    int *scanned;
    int *array_1;
    int *array_2;
    int *seg_reduce_indices;
    int *seg_reduce_result;
    int *reduced_scan;
    int *frequencies;
    int *node_states;
    int *shifts;
    MemoryAPI::allocate_device_array(&new_ptr, vertices_count + 1);
    MemoryAPI::allocate_device_array(&array_1, edges_count + 1);
    MemoryAPI::allocate_device_array(&array_2, edges_count +1);
    MemoryAPI::allocate_device_array(&seg_reduce_indices, edges_count + 1);
    MemoryAPI::allocate_device_array(&seg_reduce_result, vertices_count);
    MemoryAPI::allocate_device_array(&gathered_labels, edges_count + 1);
    MemoryAPI::allocate_device_array(&node_states, vertices_count);
    MemoryAPI::allocate_device_array(&shifts, vertices_count + 1);

    frontier.set_all_active();

    // init labels and node states
    if(_gpu_active_condition_type == LabelChangedRecently)
    {
        auto init_op = [_labels, node_states] __device__(int src_id, int position_in_frontier, int connections_count)
        {
            _labels[src_id] = src_id;
            node_states[src_id] = 4;
        };
        graph_API.compute(_graph, frontier, init_op);
    }
    else
    {
        auto init_op = [_labels, node_states] __device__(int src_id, int position_in_frontier, int connections_count)
        {
            _labels[src_id] = src_id;
            node_states[src_id] = LP_BOUNDARY_ACTIVE;
        };
        graph_API.compute(_graph, frontier, init_op);
    }

    _iterations_count = 0;

    int *updated;
    cudaMallocManaged((void**)&updated,  sizeof(int));

    do
    {
        updated[0] = 0;

        // generate new frontier with only active nodes
        if(_gpu_active_condition_type == LabelChangedRecently)
        {
            auto node_is_active = [node_states] __device__ (int src_id)->int
            {
                if(node_states[src_id] > 0)
                    return IN_FRONTIER_FLAG;
                else
                    return NOT_IN_FRONTIER_FLAG;
            };
            graph_API.generate_new_frontier(_graph, frontier, node_is_active);
        }
        else if(_gpu_active_condition_type == AlwaysActive)
        {
            frontier.set_all_active();
        }
        else
        {
            auto node_is_active = [node_states] __device__ (int src_id)->int
            {
                if(node_states[src_id] == LP_BOUNDARY_ACTIVE)
                    return IN_FRONTIER_FLAG;
                else
                    return NOT_IN_FRONTIER_FLAG;
            };
            graph_API.generate_new_frontier(_graph, frontier, node_is_active);
        }
        //print_active_percentages(node_states, vertices_count); // debug part

        // calculate shifts for gathered labels
        cudaMemset(shifts, 0, vertices_count*sizeof(int));
        int frontier_size = frontier.size();
        auto copy_degrees = [shifts, frontier_size] __device__(int src_id, int position_in_frontier, int connections_count)
        {
            shifts[position_in_frontier] = connections_count;
        };

        graph_API.compute(_graph, frontier, copy_degrees);
        thrust::exclusive_scan(thrust::device, shifts, shifts + frontier.size() + 1, shifts);

        auto gather_edge_op = [_labels, gathered_labels, shifts] __device__(int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int position_in_frontier)
        {
            int dst_label = __ldg(&_labels[dst_id]);
            int src_label = _labels[src_id];
            gathered_labels[shifts[position_in_frontier] + local_edge_pos] = dst_label;
        };

        //Gathering labels of adjacent vertices
        graph_API.advance(_graph, frontier, gather_edge_op);

        int new_vertices_count = frontier.size();
        int new_edges_count = 0;
        SAFE_CALL(cudaMemcpy(&new_edges_count, shifts + new_vertices_count , sizeof(int), cudaMemcpyDeviceToHost));

        dim3 block_edges(1024);
        dim3 grid_edges((new_edges_count - 1) / block_edges.x + 1);

        if(new_edges_count == 0)
        {
            _iterations_count++;
            continue;
        }

        SAFE_KERNEL_CALL((fill_indices<<<grid_edges, block_edges>>>(seg_reduce_indices, new_edges_count)));

        //Sorting labels of adjacent vertices in per-vertice components.
        tmp_work_buffer_for_seg_sort = array_1;

        mgpu::segmented_sort(gathered_labels, tmp_work_buffer_for_seg_sort, new_edges_count, shifts, new_vertices_count,
                             mgpu::less_t<int>(), context);

        label_differences = array_2;

        //Puts a 1 in the last element of each segment in boundaries_array. Segments are passed by v_array
        auto label_differences_initial_op = [shifts, label_differences] __device__(int src_id, int position_in_frontier, int connections_count)
        {
            long int position = shifts[position_in_frontier];
            if(position_in_frontier != 0)
            {
                label_differences[position - 1] = 1;
            }
        };
        SAFE_CALL((cudaMemset(label_differences, 0, (size_t)(sizeof(int)) * new_edges_count))); // set zeroes to difference array
        graph_API.compute(_graph, frontier, label_differences_initial_op);

        SAFE_KERNEL_CALL((label_differences_advanced <<< grid_edges, block_edges >>>
                                (label_differences, gathered_labels, new_edges_count)));

        scanned = array_1;
        //exclusive scan in order to pass repeated labels and divide different labels
        thrust::exclusive_scan(thrust::device, label_differences, label_differences + new_edges_count + 1, scanned, 0);

        int reduced_size = 0;
        SAFE_CALL(cudaMemcpy(&reduced_size, scanned + new_edges_count , sizeof(int), cudaMemcpyDeviceToHost));

        reduced_scan = array_2;
        SAFE_KERNEL_CALL((count_labels <<< grid_edges, block_edges >>> (scanned, new_edges_count, reduced_scan)));

        //new_ptr array contains new bounds of segments by getting them from scan
        //This is necessary due to shortened size of reduced_scan
        auto new_boundaries_op = [shifts, scanned, new_ptr] __device__(int src_id, int position_in_frontier, int connections_count)
        {
            new_ptr[position_in_frontier] = scanned[shifts[position_in_frontier]];
        };

        int position = 0;
        SAFE_CALL(cudaMemcpy(&position, shifts + new_vertices_count , sizeof(int), cudaMemcpyDeviceToHost));
        SAFE_CALL(cudaMemcpy(new_ptr + new_vertices_count, scanned + position, sizeof(int), cudaMemcpyDeviceToDevice));
        //new_ptr[new_vertices_count] = scanned[shifts[new_vertices_count]]; // TODO FIX without managed
        graph_API.compute(_graph, frontier, new_boundaries_op);

        frequencies = array_1;
        SAFE_KERNEL_CALL((frequency_count <<< grid_edges, block_edges >>> (frequencies, reduced_scan, reduced_size)));

        int init = REDUCE_INITIAL;
        auto seg_reduce_op =[frequencies, reduced_size] MGPU_DEVICE(int a, int b) ->int
        {
            int w_a = -1;
            int w_b = -1;
            if(a >= 0)
                w_a = frequencies[a];
            if(b >= 0)
                w_b = frequencies[b];

            if (w_a > w_b)
            {
                return a;
            }
            else
            {
                return b;
            }
        };

        //Searching for maximum frequency in each per-vertice segment
        mgpu::segreduce(seg_reduce_indices, reduced_size, new_ptr, new_vertices_count, seg_reduce_result,
                        seg_reduce_op, (int) init, context);

        if(_gpu_active_condition_type == AlwaysActive)
        {
            always_active(_graph, graph_API, frontier, new_ptr, gathered_labels,
                          _labels, updated, node_states, seg_reduce_result,
                          reduced_scan, seg_reduce_indices, _iterations_count);
        }
        else if(_gpu_active_condition_type == ActivePassiveInner)
        {
            active_passive_inner(_graph, graph_API, frontier, new_ptr, gathered_labels,
                                 _labels, updated, node_states, seg_reduce_result,
                                 reduced_scan, seg_reduce_indices, _iterations_count);
        }
        else if(_gpu_active_condition_type == LabelChangedOnPreviousIteration)
        {
            label_changed_on_previous_iteration(_graph, graph_API, frontier, new_ptr, gathered_labels,
                                                _labels, updated, node_states, seg_reduce_result,
                                                reduced_scan, seg_reduce_indices, _iterations_count);
        }
        else if(_gpu_active_condition_type == LabelChangedRecently)
        {
            label_changed_recently(_graph, graph_API, frontier, new_ptr, gathered_labels,
                                   _labels, updated, node_states, seg_reduce_result,
                                   reduced_scan, seg_reduce_indices, _iterations_count);
        }
        
        _iterations_count++;
        updated[0] = 1;
    }
    while((_iterations_count < _max_iterations) && (updated[0] > 0));

    MemoryAPI::free_device_array(new_ptr);
    MemoryAPI::free_device_array(array_1);
    MemoryAPI::free_device_array(array_2);
    MemoryAPI::free_device_array(seg_reduce_indices);
    MemoryAPI::free_device_array(seg_reduce_result);
    MemoryAPI::free_device_array(gathered_labels);
    MemoryAPI::free_device_array(node_states);
    MemoryAPI::free_device_array(shifts);

    cudaFree(updated);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void gpu_lp_wrapper<int, float>(UndirectedCSRGraph<int, float> &_graph, int *_labels, int &_iterations_count,
                                         GpuActiveConditionType _gpu_active_condition_type, int _max_iterations);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
