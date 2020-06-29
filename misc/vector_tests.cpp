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

#define BUFFER_SIZE 256

void test_copy_if()
{
    int size = 1024*1024*8;
    int *in_data = new int[size];
    int *out_data = new int[size];

    int max_buffer_size = size / (VECTOR_LENGTH * MAX_SX_AURORA_THREADS) + 1;

    cout << "MAX BUFFER SIZE: " << max_buffer_size << endl;

    int *buffers = new int[VECTOR_LENGTH * max_buffer_size];

    #pragma omp parallel for
    for(int i = 0; i < size; i++)
    {
        in_data[i] = rand()%4;
        out_data[i] = 0;
    }

    double t1 = omp_get_wtime();
    #pragma omp parallel
    {
        int reg_ptrs[VECTOR_LENGTH];

        #pragma _NEC vreg(reg_ptrs)

        #pragma _NEC vector
        for (int i = 0; i < VECTOR_LENGTH; i++)
        {
            reg_ptrs[i] = 0;
        }

        #pragma omp for
        for (int vec_start = 0; vec_start < size; vec_start += VECTOR_LENGTH)
        {
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                int val = in_data[vec_start + i];

                if(val == 1)
                {
                    int dst_idx = reg_ptrs[i] + i * max_buffer_size;
                    buffers[dst_idx] = val;
                    reg_ptrs[i]++;
                }
            }
        }

        int dump_sizes[VECTOR_LENGTH];

        for (int i = 0; i < VECTOR_LENGTH; i++)
        {
            dump_sizes[i] = reg_ptrs[i];
            reg_ptrs[i] = 0;
        }

        int out_pos = 0;
        #pragma omp parallel
        for(int reg_pos = 0; reg_pos < VECTOR_LENGTH; reg_pos++)
        {
            #pragma _NEC ivdep
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for (int i = 0; i < dump_sizes[reg_pos]; i++)
            {
                out_data[out_pos + i] = buffers[i + reg_pos * BUFFER_SIZE];
            }
            out_pos += dump_sizes[reg_pos];
        }
    }
    double t2 = omp_get_wtime();

    cout << "BW: " << 2.0 * size * sizeof(int) / ((t2 - t1) * 1e9) << " GB/s" << endl;

    delete []in_data;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    /*test_vector_length();
    test_latency();
    test_stride();*/

    test_copy_if();

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////