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
    if( i < vertices_count) {
        long int position = v_array[i];
        if (i != 0) {
            boundaries[position - 1] = 1;
        } else {
            boundaries[edges_count - 1] = 1;
        }
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
    if ((i < edges_count - 1) && (scanned_array[i + 1] != scanned_array[i])) {
        S_array[scanned_array[i]] = i;
    }
}

__global__ void new_boundaries(int *scanned_array, long long int *v_array, int vertices_count, int *S_ptr) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < vertices_count) {
        S_ptr[i] = scanned_array[v_array[i]];
    }
}


__global__ void frequency_count(int *W_array, int *S, long long int reduced_size) {
    long long int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < reduced_size)
    {
        if ((i > 0) && (S[i] != 0)) {
            W_array[i] = S[i] - S[i - 1];
        } else {
            W_array[0] = S[0] + 1;
        }
    }
}

__global__ void get_labels(int *I, int *S, int *L, int *_labels,int vertices_count) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < vertices_count)
    {
        _labels[i] = L[S[I[i]]];
    }
}

__global__ void fill_indices(int *I, long long edges_count) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < edges_count)
    {
        I[i] = i;
    }
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
    int *values;
    int *s_ptr_array;
    int *F_mem;
    int *F_scanned;
    int *I_mem;
    int *out;
    SAFE_CALL((cudaMalloc((void **) &F_mem, (size_t)(sizeof(int)) * edges_count)));
    SAFE_CALL((cudaMalloc((void **) &values, (size_t)(sizeof(int)) * edges_count)));
    SAFE_CALL((cudaMalloc((void **) &s_ptr_array, (size_t)(sizeof(long long int)) * vertices_count)));
    SAFE_CALL((cudaMalloc((void **) &F_scanned, (size_t)(sizeof(int)) * edges_count)));
    SAFE_CALL((cudaMalloc((void **) &I_mem, (size_t)(sizeof(int)) * edges_count)));
    SAFE_CALL((cudaMalloc((void **) &out, (size_t)(sizeof(long long int)) * vertices_count)));
    {
        dim3 block(1024, 1);
        dim3 grid((edges_count - 1) / block.x + 1, 1);
        SAFE_KERNEL_CALL((fill_indices<<<grid,block>>>(I_mem,edges_count))); //fill 1 in bounds
    }
    mgpu::standard_context_t context;
//    mgpu::mem_t<int> out(vertices_count, context);
    //mgpu::mem_t<int> I_mem(edges_count, context);
    MemoryAPI::allocate_device_array(&gathered_labels, edges_count);

    frontier.set_all_active();

    auto init_op =[_labels] __device__(int
    src_id, int
    connections_count) {
        _labels[src_id] = src_id;
    };

    graph_API.compute(_graph, frontier, init_op);

    _iterations_count = 0;
    while (_iterations_count < 1) // for example we can do only 1 iteration
    {
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


        mgpu::segmented_sort(gathered_labels, values, edges_count, outgoing_ptrs, vertices_count,
                             mgpu::less_t<int>(), context);
        cout<<1<<endl;

        SAFE_CALL((cudaMemset(F_mem, 0, (size_t)(sizeof(int)) * edges_count))); //was taken from group of memcpy
        {
            dim3 block(1024, 1);
            dim3 grid((vertices_count - 1) / block.x + 1, 1);
            SAFE_KERNEL_CALL(
                    (extract_boundaries_initial << < grid, block >> >
                                                           (F_mem, outgoing_ptrs, edges_count,vertices_count))); //fill 1 in bounds
        }
        cout<<2<<endl;

        {
            dim3 block(1024, 1);
            dim3 grid((edges_count - 1) / block.x + 1, 1);

            SAFE_KERNEL_CALL(
                    (extract_boundaries_optional << < grid, block >> >
                                                            (F_mem, gathered_labels, edges_count))); //sub(i+1, i)
        }

        cout<<3<<endl;
        mgpu::scan(F_mem, edges_count, F_scanned, context);
        cout<<4<<endl;
        long long int reduced_size = 0;
        int *scanned_data_ptr = F_scanned;
        SAFE_CALL(cudaMemcpy(&reduced_size, scanned_data_ptr + (edges_count - 1), sizeof(int), cudaMemcpyDeviceToHost));

        //SAFE_CALL(cudaMemcpy(&reduced_size, &F_scanned[edges_count - 1], sizeof(long long int), cudaMemcpyDeviceToHost));
        mgpu::mem_t<int> s_array(reduced_size, context);
        cout<<5<<endl;
        {
            dim3 block(1024, 1);
            dim3 grid((edges_count - 1) / block.x + 1, 1);
            SAFE_KERNEL_CALL(
                    (count_labels << < grid, block >> > (F_scanned, edges_count, s_array.data())));
        }
        cout<<6<<endl;
        {
            dim3 block(1024, 1);
            dim3 grid((vertices_count - 1) / block.x + 1, 1);
            SAFE_KERNEL_CALL((new_boundaries << < grid, block >> >
                                                        (F_scanned, outgoing_ptrs, vertices_count, s_ptr_array)));
        }
        cout<<7<<endl;
        mgpu::mem_t<int> w_array(reduced_size, context);
        {
            dim3 block(1024, 1);
            dim3 grid((edges_count - 1) / block.x + 1, 1);


            SAFE_KERNEL_CALL((frequency_count << < grid, block >> > (w_array.data(), s_array.data(),reduced_size)));
        }

        cout<<8<<endl;
        int init = 0;
        int *w_ptr = w_array.data();

        cout<<8.5<<endl;

        auto my_cool_lambda =[w_ptr] MGPU_DEVICE(int
        a, int
        b) ->int{
                if ( w_ptr[a] > w_ptr[b]){
                    return a;
                } else{
                    return b;
                }
        };


        mgpu::segreduce(I_mem, reduced_size, s_ptr_array, vertices_count, out,
                        my_cool_lambda, (int) init, context);
        cout<<9<<endl;
        {
            dim3 block(1024, 1);
            dim3 grid((edges_count - 1) / block.x + 1, 1);
            SAFE_KERNEL_CALL((get_labels << < grid, block >> >
                                                    (out, s_array.data(), gathered_labels, _labels,vertices_count)));
        }
        cout<<10<<endl;
        _iterations_count++;
    }
    SAFE_CALL(cudaFree(F_mem));
    SAFE_CALL(cudaFree(values));
    SAFE_CALL(cudaFree(F_scanned));
    SAFE_CALL(cudaFree(s_ptr_array));
    MemoryAPI::free_device_array(gathered_labels);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void gpu_lp_wrapper<int, float>(ExtendedCSRGraph<int, float> &_graph, int *_labels, int &_iterations_count);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////