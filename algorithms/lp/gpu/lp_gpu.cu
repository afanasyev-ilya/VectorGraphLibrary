#pragma once
#define REDUCE_INITIAL -1

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../../graph_processing_API/gpu/cuda_API_include.h"
#include "../../../external_libraries/moderngpu/src/moderngpu/kernel_segsort.hxx"
#include "../../../external_libraries/moderngpu/src/moderngpu/memory.hxx"
#include "../../../external_libraries/moderngpu/src/moderngpu/kernel_segreduce.hxx"
#include "../../../external_libraries/moderngpu/src/moderngpu/kernel_scan.hxx"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Puts a 1 in the last element of each segment in boundaries_array. Segments are passed by v_array
__global__ void label_differences_initial(int *differences, long long int *v_array, int vertices_count)
{
    long int i = threadIdx.x + blockIdx.x * blockDim.x;
    if( i < vertices_count + 1)
    {
        long int position = v_array[i];
        if (i != 0)
        {
            differences[position - 1] = 1;
        }
    }
}

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

//new_ptr array contains new bounds of segments by getting them from scan
//This is necessary due to shortened size of reduced_scan
__global__ void new_boundaries(int *scanned_array, long long int *v_array, int vertices_count, int *new_ptr)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < vertices_count + 1)
    {
        new_ptr[i] = scanned_array[v_array[i]];
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

template <typename _T>
void print_data(string _name, _T *_data, int _size)
{
    /*cout << _name << ": ";
    for(int i = 0; i < _size; i++)
    {
        cout << _data[i] << " ";
    }
    cout << endl << endl;*/
}

template <typename DataType, typename SegmentType>
void print_segmented_array(string _name, DataType *_data, SegmentType *_segments, int _segment_count, int _data_size)
{
    /*cout << _name << ": ";
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
        cout << " [";
        for(int i = start; i < end; i++)
        {
            if(i != (end - 1))
                cout << _data[i] << " ";
            else
                cout << _data[i];
        }
        cout << "] ";
    }
    cout << endl << endl;*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename _TVertexValue, typename _TEdgeWeight>
void gpu_lp_wrapper(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                    int *_labels,
                    int &_iterations_count)
{
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    GraphPrimitivesGPU graph_API;
    FrontierGPU frontier(_graph.get_vertices_count());

    mgpu::standard_context_t context;

    int *gathered_labels;
    int *tmp_work_buffer_for_seg_sort;
    int *s_ptr_array;
    int *label_differences;
    int *scanned;
    int *seg_reduce_indices;
    int *seg_reduce_result;
    int *reduced_scan;
    int *frequencies;
    MemoryAPI::allocate_device_array(&label_differences, edges_count + 1);
    MemoryAPI::allocate_device_array(&tmp_work_buffer_for_seg_sort, edges_count + 1);
    MemoryAPI::allocate_device_array(&s_ptr_array, vertices_count + 1);
    MemoryAPI::allocate_device_array(&scanned, edges_count + 1);
    MemoryAPI::allocate_device_array(&seg_reduce_indices, edges_count);
    MemoryAPI::allocate_device_array(&seg_reduce_result, vertices_count);
    MemoryAPI::allocate_device_array(&reduced_scan, edges_count);
    MemoryAPI::allocate_device_array(&frequencies, edges_count);
    MemoryAPI::allocate_device_array(&gathered_labels, edges_count);

    frontier.set_all_active();

    auto init_op =[_labels] __device__(int src_id, int connections_count)
    {
        _labels[src_id] = src_id;
    };

    graph_API.compute(_graph, frontier, init_op);

    _iterations_count = 0;

    int *updated;
    cudaMallocManaged((void**)&updated,  sizeof(int));
    //MemoryAPI::allocate_device_array(&updated, 1);

    dim3 block_vertices(1024);
    dim3 grid_vertices((vertices_count - 1) / block_vertices.x + 1);

    dim3 block_edges(1024);
    dim3 grid_edges((edges_count - 1) / block_edges.x + 1);

    SAFE_KERNEL_CALL((fill_indices<<<grid_edges, block_edges>>>(seg_reduce_indices, edges_count)));

    do
    {
        print_data("before iteration labels: ",  _labels, vertices_count);

        auto gather_edge_op = [_labels, gathered_labels] __device__(int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos)
        {
            int dst_label = __ldg(&_labels[dst_id]);
            gathered_labels[global_edge_pos] = dst_label;
        };
        //Gathering labels of adjacent vertices
        graph_API.advance(_graph, frontier, gather_edge_op);

        print_segmented_array("gathered labels", gathered_labels, outgoing_ptrs, vertices_count, edges_count);

        //Sorting labels of adjacent vertices in per-vertice components.
        mgpu::segmented_sort(gathered_labels, tmp_work_buffer_for_seg_sort, edges_count, outgoing_ptrs, vertices_count,
                             mgpu::less_t<int>(), context);

        print_segmented_array("gathered labels after sort", gathered_labels, outgoing_ptrs, vertices_count, edges_count);

        SAFE_CALL((cudaMemset(label_differences, 0, (size_t)(sizeof(int)) * edges_count))); //was taken from group of memcpy

        SAFE_KERNEL_CALL((label_differences_initial <<< grid_vertices, block_vertices >>>
                                                           (label_differences, outgoing_ptrs, vertices_count)));

        print_segmented_array("label_differences_initial", label_differences, outgoing_ptrs, vertices_count, edges_count);

        SAFE_KERNEL_CALL((label_differences_advanced <<< grid_edges, block_edges >>>
                                (label_differences, gathered_labels, edges_count)));

        print_segmented_array("label_differences", label_differences, outgoing_ptrs + 1, vertices_count, edges_count);

        //exclusive scan in order to pass repeated labels and divide different labels
        thrust::exclusive_scan(thrust::device, label_differences, label_differences + edges_count + 1, scanned, 0); // in-place scan

        print_segmented_array("label_differences after scan", label_differences, outgoing_ptrs, vertices_count, edges_count + 1);

        print_segmented_array("scanned after scan", scanned, outgoing_ptrs, vertices_count, edges_count);

        int reduced_size = 0;
        //int *scanned_data_ptr = scanned;
        //reduced_size = scanned[edges_count - 1];
        SAFE_CALL(cudaMemcpy(&reduced_size, scanned + edges_count , sizeof(int), cudaMemcpyDeviceToHost));
        //SAFE_CALL(cudaFree(scanned_data_ptr));
        
        SAFE_KERNEL_CALL((count_labels <<< grid_edges, block_edges >>> (scanned, edges_count, reduced_scan)));

        print_data("count labels/reduced_scan", reduced_scan, reduced_size);

        SAFE_KERNEL_CALL((new_boundaries <<< grid_vertices, block_vertices >>>
                                             (scanned, outgoing_ptrs, vertices_count, s_ptr_array)));

        print_data("outgoing ptrs", outgoing_ptrs, vertices_count + 1);
        print_data("s_ptr_array", s_ptr_array, vertices_count + 1);

        SAFE_KERNEL_CALL((frequency_count << < grid_edges, block_edges >> > (frequencies, reduced_scan, reduced_size)));

        print_data("frequencies", frequencies, reduced_size);

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

        print_data("seg_reduce_indices: ", seg_reduce_indices, edges_count);

        //Searching for maximum frequency in each per-vertice segment
        mgpu::segreduce(seg_reduce_indices, reduced_size + 1, s_ptr_array, vertices_count, seg_reduce_result,
                        seg_reduce_op, (int) init, context);

        print_data("seg_reduce_result: ",  seg_reduce_result, vertices_count);

        print_data("gathered labels: ",  gathered_labels, edges_count);
        print_data("seg_array: ",  reduced_scan, vertices_count);

        updated[0] = 0;

        //SAFE_KERNEL_CALL((get_labels <<< grid_vertices, block_vertices >>>
        //                              (seg_reduce_result, reduced_scan, gathered_labels, _labels, vertices_count, updated)));

        auto get_labels_op = [seg_reduce_result, reduced_scan, gathered_labels, _labels, updated] __device__(int src_id, int connections_count)
        {
            if((seg_reduce_result[src_id] != -1) && (_labels[src_id] != gathered_labels[reduced_scan[seg_reduce_result[src_id]]]))
            {
                _labels[src_id] = gathered_labels[reduced_scan[seg_reduce_result[src_id]]];
                updated[0] = 1;
            }
        };
        graph_API.compute(_graph, frontier, get_labels_op);

        print_data("after iteration labels: ",  _labels, vertices_count);

        _iterations_count++;
    }
    while((_iterations_count < 10) && (updated[0] > 0));

    cout << "done " << _iterations_count << " iterations" << endl;

    MemoryAPI::free_device_array(label_differences);
    MemoryAPI::free_device_array(tmp_work_buffer_for_seg_sort);
    MemoryAPI::free_device_array(s_ptr_array);
    MemoryAPI::free_device_array(scanned);
    MemoryAPI::free_device_array(seg_reduce_indices);
    MemoryAPI::free_device_array(seg_reduce_result);
    MemoryAPI::free_device_array(reduced_scan);
    MemoryAPI::free_device_array(frequencies);

    MemoryAPI::free_device_array(gathered_labels);

    // TODO manual free fro update
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void gpu_lp_wrapper<int, float>(ExtendedCSRGraph<int, float> &_graph, int *_labels, int &_iterations_count);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
