#pragma once
#define REDUCE_INITIAL -1
#define DECISION_BOUND 0.2

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 3.0

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <random>
#include <curand.h>
#include <curand_kernel.h>
#include "../../../graph_processing_API/gpu/cuda_API_include.h"
#include "../../../external_libraries/moderngpu/src/moderngpu/kernel_segsort.hxx"
#include "../../../external_libraries/moderngpu/src/moderngpu/memory.hxx"
#include "../../../external_libraries/moderngpu/src/moderngpu/kernel_segreduce.hxx"
#include "../../../external_libraries/moderngpu/src/moderngpu/kernel_scan.hxx"

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
        } else {
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

template<typename _TVertexValue, typename _TEdgeWeight>
void gpu_lp_wrapper(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                    int *_labels,
                    int &_iterations_count,
                    int _max_iterations)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
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
    int *old_labels;
    MemoryAPI::allocate_device_array(&new_ptr, vertices_count + 1);
    MemoryAPI::allocate_device_array(&array_1, edges_count + 1);
    MemoryAPI::allocate_device_array(&array_2, edges_count +1);
    MemoryAPI::allocate_device_array(&seg_reduce_indices, edges_count + 1);
    MemoryAPI::allocate_device_array(&seg_reduce_result, vertices_count);
    MemoryAPI::allocate_device_array(&gathered_labels, edges_count + 1);
    MemoryAPI::allocate_device_array(&old_labels, vertices_count);

    frontier.set_all_active();

    auto init_op =[_labels] __device__(int src_id, int connections_count)
    {
        _labels[src_id] = src_id;
    };

    graph_API.compute(_graph, frontier, init_op);

    _iterations_count = 0;

    int *updated;
    cudaMallocManaged((void**)&updated,  sizeof(int));

    dim3 block_edges(1024);
    dim3 grid_edges((edges_count - 1) / block_edges.x + 1);

    SAFE_KERNEL_CALL((fill_indices<<<grid_edges, block_edges>>>(seg_reduce_indices, edges_count)));

    do
    {
        auto gather_edge_op = [_labels, gathered_labels] __device__(int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos)
        {
            int dst_label = __ldg(&_labels[dst_id]);
            gathered_labels[global_edge_pos] = dst_label;
        };

        //Gathering labels of adjacent vertices
        graph_API.advance(_graph, frontier, gather_edge_op);

        //Sorting labels of adjacent vertices in per-vertice components.
        tmp_work_buffer_for_seg_sort = array_1;
        mgpu::segmented_sort(gathered_labels, tmp_work_buffer_for_seg_sort, edges_count, outgoing_ptrs, vertices_count,
                             mgpu::less_t<int>(), context);


        label_differences = array_2;
        SAFE_CALL((cudaMemset(label_differences, 0, (size_t)(sizeof(int)) * edges_count))); //was taken from group of memcpy

        //Puts a 1 in the last element of each segment in boundaries_array. Segments are passed by v_array
        auto label_differences_initial_op = [outgoing_ptrs, label_differences] __device__(int src_id, int connections_count)
        {
            long int position = outgoing_ptrs[src_id];
            if(src_id != 0)
            {
                label_differences[position - 1] = 1;
            }
        };
        graph_API.compute(_graph, frontier, label_differences_initial_op);

        SAFE_KERNEL_CALL((label_differences_advanced <<< grid_edges, block_edges >>>
                                (label_differences, gathered_labels, edges_count)));

        scanned = array_1;
        //exclusive scan in order to pass repeated labels and divide different labels
        thrust::exclusive_scan(thrust::device, label_differences, label_differences + edges_count + 1, scanned, 0);

        int reduced_size = 0;
        SAFE_CALL(cudaMemcpy(&reduced_size, scanned + edges_count , sizeof(int), cudaMemcpyDeviceToHost));

        reduced_scan = array_2;
        SAFE_KERNEL_CALL((count_labels <<< grid_edges, block_edges >>> (scanned, edges_count, reduced_scan)));

        //new_ptr array contains new bounds of segments by getting them from scan
        //This is necessary due to shortened size of reduced_scan
        auto new_boundaries_op = [outgoing_ptrs, scanned, new_ptr] __device__(int src_id, int connections_count)
        {
            new_ptr[src_id] = scanned[outgoing_ptrs[src_id]];
        };
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
        mgpu::segreduce(seg_reduce_indices, reduced_size + 1, new_ptr, vertices_count, seg_reduce_result,
                        seg_reduce_op, (int) init, context);

        updated[0] = 0;

        auto get_labels_op = [seg_reduce_result, reduced_scan, gathered_labels, _labels, updated , _iterations_count, old_labels] __device__(int src_id, int connections_count)
        {
            if((seg_reduce_result[src_id] != -1) && (_labels[src_id] != gathered_labels[reduced_scan[seg_reduce_result[src_id]]]))
            {
                curandState state;
                curand_init(0, 0, 0, &state);
                //if(change > 0.2)
                //{

                    int temp_label = gathered_labels[reduced_scan[seg_reduce_result[src_id]]];
                    if((_iterations_count!=0)&&(temp_label == old_labels[src_id])){
                        float change = curand_uniform(&state);
                        //printf("%f ",change);
                        if(change > 0.5){
                            old_labels[src_id] = _labels[src_id];
                            _labels[src_id] = temp_label;
                            updated[0] = 1;
                        } else{
                            //old_labels = _labels[src_id];
                            _labels[src_id] = temp_label;
                        }

                    } else {
                        old_labels[src_id] = _labels[src_id];
                        _labels[src_id] = temp_label;
                        updated[0] = 1;

                    }

                //}
            }
        };
        graph_API.compute(_graph, frontier, get_labels_op);
        _iterations_count++;
    }
    while((_iterations_count < _max_iterations) && (updated[0] > 0));

    cout << "done " << _iterations_count << " iterations" << endl;

    MemoryAPI::free_device_array(new_ptr);
    MemoryAPI::free_device_array(array_1);
    MemoryAPI::free_device_array(array_2);
    MemoryAPI::free_device_array(seg_reduce_indices);
    MemoryAPI::free_device_array(seg_reduce_result);
    MemoryAPI::free_device_array(old_labels);
    MemoryAPI::free_device_array(gathered_labels);

    cudaFree(updated);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void gpu_lp_wrapper<int, float>(ExtendedCSRGraph<int, float> &_graph, int *_labels, int &_iterations_count, int _max_iterations);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
