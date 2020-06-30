/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 4.0

#include "../graph_library.h"
#include <iostream>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void test_vector_length()
{
    int size = 1024*1024 * 16;
    int *a = new int[size];
    int *b = new int[size];

#pragma omp parallel for
    for(int i = 0; i < size; i++)
    {
        a[i] = i;
        b[i] = size - i;
    }

    vector<double> res;
    for(int vector_size = 1; vector_size <= 512; vector_size *= 2)
    {
        double t1 = omp_get_wtime();
#pragma _NEC novector
#pragma omp parallel for schedule(static, 8)
        for(int vec = 0; vec < size; vec += vector_size)
        {
#pragma _NEC vector
            for(int i = 0; i < vector_size; i++)
            {
                a[ vec+ i] += b[vec + i];
            }
        }
        double t2 = omp_get_wtime();
        cout << "bw for size " << vector_size << " is: " << 2.0 * sizeof(int) * size / ((t2 - t1) * 1e9) << " GB/s" << endl;
        res.push_back(2.0 * sizeof(int) * size / ((t2 - t1) * 1e9));
    }

    for(auto it : res)
    {
        cout << it << endl;
    }

    delete []a;
    delete []b;
}

void test_latency()
{
    int size = 1024*1024 * 8;
    int *data = new int[size];
    int *res = new int[size];
    int *ptrs1 = new int[size];
    int *ptrs2 = new int[size];
    int *ptrs3 = new int[size];

#pragma omp parallel for
    for(int i = 0; i < size; i++)
    {
        data[i] = i;
        res[i] = 0;
        ptrs1[i] = rand() % size;
        ptrs2[i] = rand() % size;
        ptrs3[i] = rand() % size;
    }

    double t1, t2;
    t1 = omp_get_wtime();
#pragma _NEC vector
#pragma omp parallel for
    for(int i = 0; i < size; i++)
    {
        res[i] = data[i];
    }
    t2 = omp_get_wtime();
    cout << "bw for size latency 1 is: " << 2.0 * sizeof(int) * size / ((t2 - t1) * 1e9) << " GB/s" << endl;

    t1 = omp_get_wtime();
#pragma _NEC vector
#pragma omp parallel for
    for(int i = 0; i < size; i++)
    {
        res[i] = data[ptrs1[i]];
    }
    t2 = omp_get_wtime();
    cout << "bw for size latency 2 is: " << 3.0 * sizeof(int) * size / ((t2 - t1) * 1e9) << " GB/s" << endl;

    t1 = omp_get_wtime();
#pragma _NEC vector
#pragma omp parallel for
    for(int i = 0; i < size; i++)
    {
        res[i] = data[ptrs2[ptrs1[i]]];
    }
    t2 = omp_get_wtime();
    cout << "bw for size latency 3 is: " << 4.0 * sizeof(int) * size / ((t2 - t1) * 1e9) << " GB/s" << endl;

    t1 = omp_get_wtime();
#pragma _NEC vector
#pragma omp parallel for
    for(int i = 0; i < size; i++)
    {
        res[i] = data[ptrs3[ptrs2[ptrs1[i]]]];
    }
    t2 = omp_get_wtime();
    cout << "bw for size latency 4 is: " << 5.0 * sizeof(int) * size / ((t2 - t1) * 1e9) << " GB/s" << endl;

    delete []data;
    delete []res;
    delete []ptrs1;
    delete []ptrs2;
    delete []ptrs3;
}

void test_stride()
{
    int size = 1024*1024 * 16;
    int *a = new int[size];
    int *b = new int[size];

#pragma omp parallel for
    for(int i = 0; i < size; i++)
    {
        a[i] = i;
        b[i] = size - i;
    }

    vector<double> res;
    for(int vector_size = 1; vector_size <= 256; vector_size *= 2)
    {
        double t1 = omp_get_wtime();
        #pragma omp parallel for
        for(int vec = 0; vec < size - 512; vec += vector_size)
        {
            #pragma _NEC vector
            for(int i = 0; i < vector_size; i++)
            {
                a[vec + i + vector_size] += b[vec + i + vector_size];
            }
        }
        double t2 = omp_get_wtime();
        cout << "bw for stride " << vector_size << " is: " << 2.0 * sizeof(int) * size / ((t2 - t1) * 1e9) << " GB/s" << endl;
        res.push_back(2.0 * sizeof(int) * size / ((t2 - t1) * 1e9));
    }

    for(auto it : res)
    {
        cout << it << endl;
    }

    delete []a;
    delete []b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define DESIRED_VALUE 1

void test_copy_if(int k_size, int sparsity_k)
{
    // preparation part
    int size = 1024*1024*k_size;
    int *in_data = new int[size];
    int *out_data = new int[size];

    int max_buffer_size = size / (VECTOR_LENGTH * MAX_SX_AURORA_THREADS) + 1;

    cout << "MAX BUFFER SIZE: " << max_buffer_size << endl;

    int *buffers = new int[VECTOR_LENGTH * max_buffer_size * MAX_SX_AURORA_THREADS];

    int check_elements_count = 0;
    #pragma omp parallel for reduction(+: check_elements_count)
    for(int i = 0; i < size; i++)
    {
        in_data[i] = rand()%sparsity_k;
        out_data[i] = 0;
        if(in_data[i] == 1)
        {
            check_elements_count += 1;
        }
        if(i < VECTOR_LENGTH * max_buffer_size * MAX_SX_AURORA_THREADS)
            buffers[i] = 5;
    }
    cout << check_elements_count << " elements should be copied " << endl;

    // copy if started
    double t1 = omp_get_wtime();
    int output_size = 0;
    int shifts_array[MAX_SX_AURORA_THREADS];
    #pragma omp parallel num_threads(MAX_SX_AURORA_THREADS) shared(output_size)
    {
        int tid = omp_get_thread_num();
        int *private_buffer = &buffers[VECTOR_LENGTH * max_buffer_size * tid];

        int reg_ptrs[VECTOR_LENGTH];

        #pragma _NEC vreg(reg_ptrs)

        #pragma _NEC vector
        for (int i = 0; i < VECTOR_LENGTH; i++)
        {
            reg_ptrs[i] = 0;
        }

        // copy data to buffers
        #pragma omp for schedule(static)
        for (int vec_start = 0; vec_start < size; vec_start += VECTOR_LENGTH)
        {
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                int val = in_data[vec_start + i];

                if(val == DESIRED_VALUE)
                {
                    int dst_buffer_idx = reg_ptrs[i] + i * max_buffer_size;
                    private_buffer[dst_buffer_idx] = vec_start + i;
                    reg_ptrs[i]++;
                }
            }
        }

        // calculate sizes
        int dump_sizes[VECTOR_LENGTH];
        #pragma _NEC vector
        for (int i = 0; i < VECTOR_LENGTH; i++)
        {
            dump_sizes[i] = reg_ptrs[i];
        }
        int private_size = 0;
        #pragma _NEC vector
        for(int reg_pos = 0; reg_pos < VECTOR_LENGTH; reg_pos++)
        {
            private_size += dump_sizes[reg_pos];
        }

        // calculate output offsets
        shifts_array[tid] = private_size;
        #pragma omp barrier
        #pragma omp master
        {
            int cur_shift = 0;
            for(int i = 1; i < MAX_SX_AURORA_THREADS; i++)
            {
                shifts_array[i] += shifts_array[i - 1];
            }
            output_size = shifts_array[MAX_SX_AURORA_THREADS - 1];
            for(int i = (MAX_SX_AURORA_THREADS - 1); i >= 1; i--)
            {
                shifts_array[i] = shifts_array[i - 1];
            }
            shifts_array[0] = 0;
        }
        #pragma omp barrier
        int output_offset = shifts_array[tid];

        // save data to output array
        int current_pos = 0;
        for(int reg_pos = 0; reg_pos < VECTOR_LENGTH; reg_pos++)
        {
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for (int i = 0; i < dump_sizes[reg_pos]; i++)
            {
                int src_buffer_idx = i + reg_pos * max_buffer_size;
                out_data[output_offset + current_pos + i] = private_buffer[src_buffer_idx];
            }
            current_pos += dump_sizes[reg_pos];
        }
    }
    double t2 = omp_get_wtime();

    // print stats
    cout << "BW: " << 2.0 * size * sizeof(int) / ((t2 - t1) * 1e9) << " GB/s" << endl;
    cout << "we saved " << output_size << " elements" << endl;

    cout << "first 20 elements of output array: " << endl;
    for(int i = 0; i < 20; i++)
        cout << out_data[i] << endl;

    for(int i = 0; i < MAX_SX_AURORA_THREADS; i++)
    {
        cout << "shift " << shifts_array[i] << endl;
    }

    // check
    int error_count = 0;
    for(int i = 0; i < output_size; i++)
    {
        int idx = out_data[i];
        if(in_data[idx] != DESIRED_VALUE)
        {
            if(error_count < 20)
                cout << in_data[idx] << " error for copied index " << idx << " in pos " << i << endl;
            error_count++;
        }
    }
    cout << "size check: " << output_size << " vs " << check_elements_count << endl;
    cout << "error count: " << error_count << endl;

    delete []in_data;
    delete []out_data;
    delete []buffers;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    /*test_vector_length();
    test_latency();
    test_stride();*/

    test_copy_if(atoi(argv[1]), atoi(argv[2]));

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////