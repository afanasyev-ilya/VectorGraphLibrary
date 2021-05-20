#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

using namespace std;

void saxpy(float a, float * __restrict z, const float * __restrict x, const float * __restrict y, size_t size)
{
    #pragma omp parallel for
    for(size_t i = 0; i < size; i++)
        z[i] = a*x[i] + y[i];
}

void gather(const int *__restrict data, const int * __restrict indexes, int * __restrict result, size_t size)
{
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < size; i++)
        result[i] = data[indexes[i]];
}

void scatter(int *__restrict data, const int * __restrict indexes, int * __restrict result, size_t size)
{
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < size; i++)
        data[indexes[i]] = result[i];
}

int main(void)
{
    size_t size = 1610612736;
    float *x, *y, *z;
    x = (float*)memalign(0x200, size *sizeof(float));
    y = (float*)memalign(0x200, size *sizeof(float));
    z = (float*)memalign(0x200, size *sizeof(float));

    #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
        z[i] = 3.0f;
    }

    double t1 = omp_get_wtime();
    saxpy(2.0f, z, x, y, size);
    double t2 = omp_get_wtime();
    cout << size * sizeof(float) * 3.0 / ((t2 - t1)*1e9) << " GB/s" << endl;

    free(x);
    free(y);
    free(z);

    int *result, *data, *indexes;
    int large_size = size;
    int small_size = 16*1024*1024;

    result = (int*)memalign(0x200, large_size *sizeof(int));
    indexes = (int*)memalign(0x200, large_size *sizeof(int));
    data = (int*)memalign(0x200, small_size *sizeof(int));
    #pragma omp parallel
    {
        unsigned int myseed = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (size_t i = 0; i < large_size; i++)
        {
            result[i] = rand_r(&myseed);
        }

        #pragma omp for schedule(static)
        for (size_t i = 0; i < small_size; i++)
        {
            data[i] = rand_r(&myseed);
        }
    }

    cout << "large size is " << large_size * sizeof(int) / (1024*1024) << " MB" << endl;
    for(int rad = 64*1024; rad <= 16*1024*1024; rad *= 2)
    {
        if(rad * sizeof(int) / (1024*1024) <= 1)
            cout << "Rad size is " << rad * sizeof(int) / (1024) << " KB" << endl;
        else
            cout << "Rad size is " << rad * sizeof(int) / (1024*1024) << " MB" << endl;

        #pragma omp parallel
        {
            unsigned int myseed = omp_get_thread_num();
            #pragma omp for schedule(static)
            for (size_t i = 0; i < large_size; i++)
            {
                indexes[i] = (int) rand_r(&myseed) % rad;
            }
        }

        t1 = omp_get_wtime();
        gather(data, indexes, result, large_size);
        t2 = omp_get_wtime();
        cout << large_size * sizeof(int) * 3.0 / ((t2 - t1)*1e9) << " GB/s -- gather" << endl;

        t1 = omp_get_wtime();
        gather(data, indexes, result, large_size);
        t2 = omp_get_wtime();
        cout << large_size * sizeof(int) * 3.0 / ((t2 - t1)*1e9) << " GB/s -- gather" << endl;

        t1 = omp_get_wtime();
        gather(data, indexes, result, large_size);
        t2 = omp_get_wtime();
        cout << large_size * sizeof(int) * 3.0 / ((t2 - t1)*1e9) << " GB/s -- gather" << endl;

        /*t1 = omp_get_wtime();
        scatter(data, indexes, result, large_size);
        t2 = omp_get_wtime();
        cout << large_size * sizeof(int) * 3.0 / ((t2 - t1)*1e9) << " GB/s -- scatter" << endl*/
        cout << endl;
    }

    free(data);
    free(indexes);
    free(result);
}