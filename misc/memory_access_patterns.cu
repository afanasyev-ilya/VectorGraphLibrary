#define INT_ELEMENTS_PER_EDGE 1.0

#include "../graph_library.h"
#include <iostream>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ alligned_kernel(int *large_data,
                                int *out_data,
                                int *borders,
                                int small_size,
                                int large_size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < large_size)
        out_data[idx] = large_data[idx];
}

void __global__ non_alligned_kernel(int *large_data,
                                int *out_data,
                                int *borders,
                                int small_size,
                                int large_size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int *start_in = &large_data[blockIdx.x * blockDim.x];
    int *start_out = &out_data[blockIdx.x * blockDim.x];
    if(idx < large_size)
        start_out[( threadIdx.x + 7)] = start_in[(threadIdx.x + 7)];
}

void __global__ time_local_kernel(int *large_data,
                                    int *out_data,
                                    int *borders,
                                    int small_size,
                                    int large_size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < small_size - 1)
    {
        int start = borders[idx];
        int end = borders[idx + 1];
        for(int j = start; j < end; j++)
        {
            out_data[j] = large_data[j];
        }
    }
}

void __global__ strided_kernel(int *large_data,
                                  int *out_data,
                                  int *borders,
                                  int small_size,
                                  int large_size,
                                  int stride)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < large_size / stride)
    {
        out_data[idx * stride] = large_data[idx * stride];
    }
}

void __global__ random_access_kernel(int *large_data,
                               int *out_data,
                               int *borders,
                               int small_size,
                               int large_size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < large_size)
    {
        out_data[idx] = large_data[borders[idx]];
    }
}

void __global__ random_access_kernel_d(double *large_data,
                                       double *out_data,
                                     int *borders,
                                     int small_size,
                                     int large_size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < large_size)
    {
        out_data[idx] = large_data[borders[idx]];
    }
}

int main(int argc, const char * argv[])
{
    double t1, t2, BW;
    cout << "GPU Memory access pattern test..." << endl;

    int coef = 32;
    int small_size = 8*1024*1024;
    int large_size = coef * small_size;

    int *large_data;
    int *out_data;
    int *borders;
    MemoryAPI::allocate_managed_array(&large_data, large_size);
    MemoryAPI::allocate_managed_array(&out_data, large_size);
    MemoryAPI::allocate_managed_array(&borders, large_size);

    for(int i = 0; i < small_size; i++)
    {
        borders[i] = i * coef;
        large_data[i] = 0;
        out_data[i] = 0;
    }

    MemoryAPI::prefetch_managed_array(large_data, large_size);
    MemoryAPI::prefetch_managed_array(out_data, large_size);
    MemoryAPI::prefetch_managed_array(borders, small_size);

    SAFE_KERNEL_CALL((time_local_kernel<<< small_size/1024, 1024 >>> (large_data, out_data, borders, small_size, large_size)));
    SAFE_KERNEL_CALL((time_local_kernel<<< small_size/1024, 1024 >>> (large_data, out_data, borders, small_size, large_size)));

    t1 = omp_get_wtime();
    SAFE_KERNEL_CALL((alligned_kernel<<< large_size/1024, 1024 >>> (large_data, out_data, borders, small_size, large_size)));
    cudaDeviceSynchronize();
    t2 = omp_get_wtime();
    BW = 2.0*(large_size)*sizeof(int) / (1e9 * (t2 - t1));
    cout << "aligned BW: " << BW << " GB/s " << 100 * BW / 720 << " %" << endl;

    t1 = omp_get_wtime();
    SAFE_KERNEL_CALL((non_alligned_kernel<<< large_size/1024 - 1, 1024 >>> (large_data, out_data, borders, small_size, large_size)));
    cudaDeviceSynchronize();
    t2 = omp_get_wtime();
    BW = 2.0*(large_size)*sizeof(int) / (1e9 * (t2 - t1));
    cout << "non_aligned BW: " << BW << " GB/s " << 100 * BW / 720 << " %" << endl;

    t1 = omp_get_wtime();
    SAFE_KERNEL_CALL((time_local_kernel<<< small_size/1024, 1024 >>> (large_data, out_data, borders, small_size, large_size)));
    cudaDeviceSynchronize();
    t2 = omp_get_wtime();
    BW = 2.0*(large_size)*sizeof(int) / (1e9 * (t2 - t1));
    cout << "time local BW: " << BW << " GB/s " << 100 * BW / 720 << " %" << endl;

    int stride = 2;
    t1 = omp_get_wtime();
    SAFE_KERNEL_CALL((strided_kernel<<< large_size/(1024*stride), 1024 >>> (large_data, out_data, borders, small_size, large_size, stride)));
    cudaDeviceSynchronize();
    t2 = omp_get_wtime();
    BW = 2.0*(large_size/stride)*sizeof(int) / (1e9 * (t2 - t1));
    cout << "stride 2 BW: " << BW << " GB/s " << 100 * BW / 720 << " %" << endl;

    stride = 4;
    t1 = omp_get_wtime();
    SAFE_KERNEL_CALL((strided_kernel<<< large_size/(1024*stride), 1024 >>> (large_data, out_data, borders, small_size, large_size, stride)));
    cudaDeviceSynchronize();
    t2 = omp_get_wtime();
    BW = 2.0*(large_size/stride)*sizeof(int) / (1e9 * (t2 - t1));
    cout << "stride 4 BW: " << BW << " GB/s " << 100 * BW / 720 << " %" << endl;

    stride = 8;
    t1 = omp_get_wtime();
    SAFE_KERNEL_CALL((strided_kernel<<< large_size/(1024*stride), 1024 >>> (large_data, out_data, borders, small_size, large_size, stride)));
    cudaDeviceSynchronize();
    t2 = omp_get_wtime();
    BW = 2.0*(large_size/stride)*sizeof(int) / (1e9 * (t2 - t1));
    cout << "stride 8 BW: " << BW << " GB/s " << 100 * BW / 720 << " %" << endl;

    stride = 128;
    t1 = omp_get_wtime();
    SAFE_KERNEL_CALL((strided_kernel<<< large_size/(1024*stride), 1024 >>> (large_data, out_data, borders, small_size, large_size, stride)));
    cudaDeviceSynchronize();
    t2 = omp_get_wtime();
    BW = 2.0*(large_size/stride)*sizeof(int) / (1e9 * (t2 - t1));
    cout << "stride 128 BW: " << BW << " GB/s " << 100 * BW / 720 << " %" << endl;

    stride = 512;
    t1 = omp_get_wtime();
    SAFE_KERNEL_CALL((strided_kernel<<< large_size/(1024*stride), 1024 >>> (large_data, out_data, borders, small_size, large_size, stride)));
    cudaDeviceSynchronize();
    t2 = omp_get_wtime();
    BW = 2.0*(large_size/stride)*sizeof(int) / (1e9 * (t2 - t1));
    cout << "stride 512 BW: " << BW << " GB/s " << 100 * BW / 720 << " %" << endl;

    for(int i = 0; i < large_size; i++)
    {
        borders[i] = rand()%(2*1024*1024 / sizeof(int));
    }
    SAFE_KERNEL_CALL((alligned_kernel<<< large_size/1024, 1024 >>> (borders, out_data, borders, small_size, large_size)));

    t1 = omp_get_wtime();
    SAFE_KERNEL_CALL((random_access_kernel<<< large_size/(1024), 1024 >>> (large_data, out_data, borders, small_size, large_size)));
    cudaDeviceSynchronize();
    t2 = omp_get_wtime();
    BW = (2.0*(large_size)*sizeof(int) + 1.0*sizeof(int)*large_size) / (1e9 * (t2 - t1));
    cout << "random 2 MB BW: " << BW << " GB/s " << 100 * BW / 720 << " %" << endl;

    for(int i = 0; i < large_size; i++)
    {
        borders[i] = rand()%small_size;
    }
    SAFE_KERNEL_CALL((alligned_kernel<<< large_size/1024, 1024 >>> (borders, out_data, borders, small_size, large_size)));

    t1 = omp_get_wtime();
    SAFE_KERNEL_CALL((random_access_kernel<<< large_size/(1024), 1024 >>> (large_data, out_data, borders, small_size, large_size)));
    cudaDeviceSynchronize();
    t2 = omp_get_wtime();
    BW = (2.0*(large_size)*sizeof(int) + 1.0*sizeof(int)*large_size) / (1e9 * (t2 - t1));
    cout << "random small size BW: " << BW << " GB/s " << 100 * BW / 720 << " %" << endl;

    t1 = omp_get_wtime();
    SAFE_KERNEL_CALL((random_access_kernel_d<<< large_size/(2*1024), 1024 >>> ((double*)large_data, (double*)out_data, borders, small_size, large_size/2)));
    cudaDeviceSynchronize();
    t2 = omp_get_wtime();
    BW = (2.0*(large_size/2)*sizeof(double) + 1.0*sizeof(int)*large_size/2) / (1e9 * (t2 - t1));
    cout << "double random 64 KB BW: " << BW << " GB/s " << 100 * BW / 720 << " %" << endl;

    for(int i = 0; i < large_size; i++)
    {
        borders[i] = rand()%(64*1024 / sizeof(int));
    }
    SAFE_KERNEL_CALL((alligned_kernel<<< large_size/1024, 1024 >>> (borders, out_data, borders, small_size, large_size)));

    t1 = omp_get_wtime();
    SAFE_KERNEL_CALL((random_access_kernel<<< large_size/(1024), 1024 >>> (large_data, out_data, borders, small_size, large_size)));
    cudaDeviceSynchronize();
    t2 = omp_get_wtime();
    BW = (2.0*(large_size)*sizeof(int) + 1.0*sizeof(int)*large_size) / (1e9 * (t2 - t1));
    cout << "random 64 KB BW: " << BW << " GB/s " << 100 * BW / 720 << " %" << endl;

    t1 = omp_get_wtime();
    SAFE_KERNEL_CALL((random_access_kernel_d<<< large_size/(2*1024), 1024 >>> ((double*)large_data, (double*)out_data, borders, small_size, large_size/2)));
    cudaDeviceSynchronize();
    t2 = omp_get_wtime();
    BW = (2.0*(large_size/2)*sizeof(double) + 1.0*sizeof(int)*large_size/2) / (1e9 * (t2 - t1));
    cout << "double random 64 KB BW: " << BW << " GB/s " << 100 * BW / 720 << " %" << endl;

    MemoryAPI::free_device_array(large_data);
    MemoryAPI::free_device_array(out_data);
    MemoryAPI::free_device_array(borders);
}
