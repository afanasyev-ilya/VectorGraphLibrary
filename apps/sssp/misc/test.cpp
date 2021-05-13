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

int main(void)
{
    size_t size = 1610612736;
    float *x, *y, *z;
    x = (float*)memalign(0x200, size *sizeof(float));
    y = (float*)memalign(0x200, size *sizeof(float));
    z = (float*)memalign(0x200, size *sizeof(float));

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
        z[i] = 3.0f;
    }

    double t1 = omp_get_wtime();
    saxpy(2.0f, z, x, y, size);
    double t2 = omp_get_wtime();
    cout << size * sizeof(float) * 3.0 / ((t2 - t1)*1e9) << " GB/s" << endl;

    t1 = omp_get_wtime();
    saxpy(2.0f, z, x, y, size);
    t2 = omp_get_wtime();
    cout << size * sizeof(float) * 3.0 / ((t2 - t1)*1e9) << " GB/s" << endl;

    free(x);
    free(y);
    free(z);
}