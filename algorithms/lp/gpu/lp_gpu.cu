#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../../graph_processing_API/gpu/cuda_API_include.h"
#include "../../../external_libraries/moderngpu/src/moderngpu/kernel_segsort.hxx"
#include "../../../external_libraries/moderngpu/src/moderngpu/memory.hxx"
#include "../../../external_libraries/moderngpu/src/moderngpu/kernel_segreduce.hxx"
#include "../../../external_libraries/moderngpu/src/moderngpu/kernel_scan.hxx"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Device functions
__global__ void extract_boundaries_initial(int *boundaries, long long int *v_array, long long int edges_count,int vertices_count) {

    long int i = threadIdx.x + blockIdx.x * blockDim.x;
    if( i < vertices_count + 1) {
        long int position = v_array[i];
        if (i != 0) {
            boundaries[position - 1] = 1;
        }// else {
        //    boundaries[edges_count - 1] = 1;
        //}
    }
}

__global__ void extract_boundaries_optional(int *boundaries, int *dest_labels, int edges_count) {
    long long int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ((boundaries[i] != 1) && (i < edges_count)) {
        if (dest_labels[i] != dest_labels[i + 1]) {
            boundaries[i] = 1;
        }
    }
}

__global__ void count_labels(int *scanned_array, long long int edges_count, int *S_array) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ((i < edges_count) && (scanned_array[i + 1] != scanned_array[i])) {
        S_array[scanned_array[i]] = i;
    }
}

__global__ void new_boundaries(int *scanned_array, long long int *v_array, int vertices_count, int *S_ptr, long long edges_count) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < vertices_count +1) {

        S_ptr[i] = scanned_array[v_array[i]];


        //printf("%d %ld %d!!!!\n", i, v_array[i], scanned_array[v_array[i]]);
    }
}


__global__ void frequency_count(int *W_array, int *S, long long int reduced_size) {
    long long int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < reduced_size)
    {
        if ((i > 0)) {
            W_array[i] = S[i] - S[i - 1];
        } else {
            W_array[0] = S[0] + 1;
        }
    }
}

__global__ void get_labels(int *out, int *s_array, int *gathered_labels, int *_labels, int vertices_count, int *_updated)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < vertices_count)
    {
        if((out[i] != -1) && (_labels[i] != gathered_labels[s_array[out[i]]]))
        {
            _labels[i] = gathered_labels[s_array[out[i]]];
            _updated[0] = 1;
        }
    }
}

__global__ void fill_indices(int *I, long long edges_count)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < edges_count)
    {
        I[i] = i;
    }
}

template <typename _T>
void print_data(string _name, _T *_data, int _size)
{
//    cout << _name << ": ";
//    for(int i = 0; i < _size; i++)
//    {
//        cout << _data[i] << " ";
//    }
//    cout << endl << endl;
}

template <typename DataType, typename SegmentType>
void print_segmented_array(string _name, DataType *_data, SegmentType *_segments, int _segment_count, int _data_size)
{
//    cout << _name << ": ";
//    for(int i = 0; i < _data_size; i++)
//    {
//        cout << _data[i] << " ";
//    }
//    cout << endl;
//
//    cout << _name << " with segments: ";
//
//    for(int segment = 0; segment < _segment_count; segment++)
//    {
//        int start = _segments[segment];
//        int end = _segments[segment + 1];
//        cout << " [";
//        for(int i = start; i < end; i++)
//        {
//            if(i != (end - 1))
//                cout << _data[i] << " ";
//            else
//                cout << _data[i];
//        }
//        cout << "] ";
//    }
//    cout << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename _TVertexValue, typename _TEdgeWeight>
void gpu_lp_wrapper(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                    int *_labels,
                    int &_iterations_count) {
    LOAD_EXTENDED_CSR_GRAPH_DATA(_graph);
    GraphPrimitivesGPU graph_API;
    FrontierGPU frontier(_graph.get_vertices_count());


    int *gathered_labels;
    int *tmp_work_buffer_for_seg_sort;
    int *s_ptr_array;
    int *F_mem;
    int *F_scanned;
    int *I_mem;
    int *seg_reduce_result;
    int *s_array;
    int *w_array;
    MemoryAPI::allocate_device_array(&F_mem, edges_count+1);
    MemoryAPI::allocate_device_array(&tmp_work_buffer_for_seg_sort, edges_count+1);
    MemoryAPI::allocate_device_array(&s_ptr_array, vertices_count+1);
    MemoryAPI::allocate_device_array(&F_scanned, edges_count + 1 +1);
    MemoryAPI::allocate_device_array(&I_mem, edges_count+1);
    MemoryAPI::allocate_device_array(&seg_reduce_result, vertices_count+1);
    MemoryAPI::allocate_device_array(&s_array, edges_count+1);
    MemoryAPI::allocate_device_array(&w_array, edges_count+1);

    {
        dim3 block(1024, 1);
        dim3 grid((edges_count - 1) / block.x + 1, 1);
        SAFE_KERNEL_CALL((fill_indices<<<grid,block>>>(I_mem,edges_count))); //fill 1 in bounds
    }

    mgpu::standard_context_t context;
    MemoryAPI::allocate_device_array(&gathered_labels, edges_count);

    frontier.set_all_active();

    auto init_op =[_labels] __device__(int src_id, int connections_count) {
        _labels[src_id] = src_id;
    };

    graph_API.compute(_graph, frontier, init_op);

    _iterations_count = 0;

    int *updated;
    MemoryAPI::allocate_device_array(&updated, 1);

    do
    {
        print_data("before iteration labels: ",  _labels, vertices_count);

        auto gather_edge_op = [_labels, gathered_labels]
                __device__(int
        src_id, int
        dst_id, int
        local_edge_pos, long long int
        global_edge_pos){
            int dst_label = __ldg(&_labels[dst_id]);
            gathered_labels[global_edge_pos] = dst_label;
        };
        graph_API.advance(_graph, frontier, gather_edge_op);

        cudaDeviceSynchronize();

        print_segmented_array("gathered labels", gathered_labels, outgoing_ptrs, vertices_count, edges_count);


        mgpu::segmented_sort(gathered_labels, tmp_work_buffer_for_seg_sort, edges_count, outgoing_ptrs, vertices_count,
                             mgpu::less_t<int>(), context);

        print_segmented_array("gathered labels after sort", gathered_labels, outgoing_ptrs, vertices_count, edges_count);


        //cout<<1<<endl;

        SAFE_CALL((cudaMemset(F_mem, 0, (size_t)(sizeof(int)) * edges_count))); //was taken from group of memcpy
        dim3 block_vertices(1024);
        dim3 grid_vertices((vertices_count - 1) / block_vertices.x + 1);
        SAFE_KERNEL_CALL((extract_boundaries_initial <<< grid_vertices, block_vertices >>>
                                                           (F_mem, outgoing_ptrs, edges_count,vertices_count))); //fill 1 in bounds
        //cout<<2<<endl;

        print_segmented_array("extract_boundaries_initial", F_mem, outgoing_ptrs, vertices_count, edges_count);

        dim3 block_edges(1024);
        dim3 grid_edges((edges_count - 1) / block_edges.x + 1);

        SAFE_KERNEL_CALL(
                    (extract_boundaries_optional <<< grid_edges, block_edges >>>
                                                            (F_mem, gathered_labels, edges_count)));

        print_segmented_array("F_mem", F_mem, outgoing_ptrs + 1, vertices_count, edges_count);

        cout<<3<<endl;
        //mgpu::scan(F_mem, edges_count + 1, F_scanned, context);
        //ScanExc(data->get(), N, &total, context);
        thrust::exclusive_scan(thrust::device, F_mem, F_mem + edges_count + 1, F_scanned, 0); // in-place scan

        print_segmented_array("F_mem after scan", F_mem, outgoing_ptrs, vertices_count, edges_count + 1);

        print_segmented_array("F_scanned after scan", F_scanned, outgoing_ptrs, vertices_count, edges_count);

        cout<<4<<endl;
        long long int reduced_size = 0;
        int *scanned_data_ptr = F_scanned;
        int t_reduced_size = 0;
        reduced_size = F_scanned[edges_count -1]; // TODO fix
        cout << "reduced size: " << reduced_size << endl;
        cout << "edges count: " << edges_count << endl;
        //SAFE_CALL(cudaMemcpy(&t_reduced_size, scanned_data_ptr + edges_count , sizeof(int), cudaMemcpyDeviceToHost));

        //SAFE_CALL(cudaMemcpy(&reduced_size, &F_scanned[edges_count - 1], sizeof(long long int), cudaMemcpyDeviceToHost));
        //mgpu::mem_t<int> s_array(reduced_size+1, context);
        cout<<5<<endl;
        SAFE_KERNEL_CALL(
                    (count_labels <<< grid_edges, block_edges >>> (F_scanned, edges_count, s_array)));
        cout<<6<<endl;
        //print_segmented_array("count_labels/s_array", s_array, outgoing_ptrs, vertices_count, edges_count);
        print_data("count labels/s_array", s_array, reduced_size);

        {
            dim3 block(1024,1);
            dim3 grid((vertices_count) / block.x + 1,1);
            SAFE_KERNEL_CALL((new_boundaries << < grid, block >> >
                                                        (F_scanned, outgoing_ptrs, vertices_count, s_ptr_array, edges_count)));
        }
        cout<<7<<endl;
        int w_size = s_ptr_array[vertices_count] -1;
        cout << "wsize: " << w_size << endl;

        print_data("outgoing ptrs", outgoing_ptrs, vertices_count + 1);
        print_data("s_ptr_array", s_ptr_array, vertices_count + 1);

        //int* ptr = s_ptr_array;
        //SAFE_CALL(cudaMemcpy(&w_size, ptr + vertices_count , sizeof(int), cudaMemcpyDeviceToHost));

        SAFE_KERNEL_CALL((frequency_count << < grid_edges, block_edges >> > (w_array, s_array, w_size)));

        print_data("w_array", w_array, reduced_size);

        cout<<8<<endl;
        int init = -1;

        cout<<8.5<<endl;

        auto seg_reduce_op =[w_array] MGPU_DEVICE(int a, int b) ->int{
            if ( w_array[a] > w_array[b])
            {
                return a;
            }
            else
            {
                return b;
            }
            return true;
        };

        print_data("I_mem: ", I_mem, edges_count);
        print_data("w_ptr: ", w_array, w_size);

        mgpu::segreduce(I_mem, reduced_size + 1 , s_ptr_array, vertices_count, seg_reduce_result,
                        seg_reduce_op, (int) init, context);

        print_data("seg_reduce_result: ",  seg_reduce_result, vertices_count);

        print_data("gathered labels: ",  gathered_labels, edges_count);
        print_data("seg_array: ",  s_array, vertices_count);

        updated[0] = 0;

        SAFE_KERNEL_CALL((get_labels << < grid_vertices, block_vertices >> >
                                                    (seg_reduce_result, s_array, gathered_labels, _labels, vertices_count, updated)));
        cout<<10<<endl;
        cout << "updated: " << updated[0] << endl;

        print_data("after iteration labels: ",  _labels, vertices_count);

        _iterations_count++;
    }
    while((_iterations_count < 10) && (updated[0] > 0));

    cout << "done " << _iterations_count << " iterations" << endl;

    MemoryAPI::free_device_array(F_mem);
    MemoryAPI::free_device_array(tmp_work_buffer_for_seg_sort);
    MemoryAPI::free_device_array(s_ptr_array);
    MemoryAPI::free_device_array(F_scanned);
    MemoryAPI::free_device_array(I_mem);
    MemoryAPI::free_device_array(seg_reduce_result);
    MemoryAPI::free_device_array(s_array);
    MemoryAPI::free_device_array(w_array);

    MemoryAPI::free_device_array(gathered_labels);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void gpu_lp_wrapper<int, float>(ExtendedCSRGraph<int, float> &_graph, int *_labels, int &_iterations_count);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
