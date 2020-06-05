#define INT_ELEMENTS_PER_EDGE 1.0

#include "../graph_library.h"
#include <iostream>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    double t1, t2;
    cout << "Memory access pattern test..." << endl;

    int coef = 32;
    int small_size = 32*1024*1024;
    int large_size = coef * small_size;

    int *large_data = new int[large_size];
    int *out_data = new int[large_size];
    int *borders = new int[small_size];

    #pragma omp parallel for
    for(int i = 0; i < small_size; i++)
    {
        borders[i] = i * coef;
        large_data[i] = 0;
        out_data[i] = 0;
    }

    double BW = 0;

    t1 = omp_get_wtime();
    #pragma _NEC vector
    #pragma omp parallel for
    for(int i = 0; i < large_size; i++)
    {
        out_data[i] = large_data[i];
    }
    t2 = omp_get_wtime();
    BW = 2.0*large_size*sizeof(int) / (1e9 * (t2 - t1));
    cout << "alligned BW: " << BW << " GB/s " << 100.0*BW/1200 << " %" << endl;

    t1 = omp_get_wtime();
    #pragma _NEC vector
    #pragma omp parallel for
    for(int i = 0; i < large_size - 10; i++)
    {
        out_data[i + 10] = large_data[i + 10];
    }
    t2 = omp_get_wtime();
    BW = 2.0*large_size*sizeof(int) / (1e9 * (t2 - t1));
    cout << "non-alligned BW: " << BW << " GB/s " << 100.0*BW/1200 << " %" << endl;

    t1 = omp_get_wtime();
    #pragma _NEC vector
    #pragma omp parallel for
    for(int i = 0; i < large_size/2; i++)
    {
        out_data[i * 2] = large_data[i * 2];
    }
    t2 = omp_get_wtime();
    BW = 2.0*(large_size/2.0)*sizeof(int) / (1e9 * (t2 - t1));
    t2 = omp_get_wtime();
    cout << "stride 2 BW: " << BW << " GB/s " << 100.0*BW/1200 << " %" << endl;

    t1 = omp_get_wtime();
    #pragma _NEC vector
    #pragma omp parallel for
    for(int i = 0; i < large_size/4; i++)
    {
        out_data[i * 4] = large_data[i * 4];
    }
    t2 = omp_get_wtime();
    BW = 2.0*(large_size/4.0)*sizeof(int) / (1e9 * (t2 - t1));
    cout << "stride 4 BW: " << BW << " GB/s " << 100.0*BW/1200 << " %" << endl;

    t1 = omp_get_wtime();
    #pragma _NEC vector
    #pragma omp parallel for
    for(int i = 0; i < large_size/8; i++)
    {
        out_data[i * 8] = large_data[i * 8];
    }
    t2 = omp_get_wtime();
    BW = 2.0*(large_size/8)*sizeof(int) / (1e9 * (t2 - t1));
    cout << "stride 8 BW: " << BW << " GB/s " << 100.0*BW/1200 << " %" << endl;

    t1 = omp_get_wtime();
    #pragma _NEC vector
    #pragma omp parallel for
    for(int i = 0; i < large_size/128; i++)
    {
        out_data[i * 128] = large_data[i * 128];
    }
    t2 = omp_get_wtime();
    BW = 2.0*(large_size/128)*sizeof(int) / (1e9 * (t2 - t1));
    cout << "stride 128 BW: " << BW << " GB/s " << 100.0*BW/1200 << " %" << endl;

    t1 = omp_get_wtime();
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp parallel for
    for(int i = 0; i < small_size - 1; i++)
    {
        int start = borders[i];
        int end = borders[i + 1];
        for(int j = start; j < end; j++)
        {
            out_data[j] = large_data[j];
        }
    }
    t2 = omp_get_wtime();
    BW = 2.0*(large_size/128)*sizeof(int) / (1e9 * (t2 - t1));
    cout << "one by one BW: " << BW << " GB/s " << 100.0*BW/1200 << " %" << endl;

    t1 = omp_get_wtime();
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma omp parallel for
    for(int i = 0; i < small_size - 1; i++)
    {
        int start = borders[i];
        int end = borders[i + 1];
        #pragma _NEC novector
        for(int j = start; j < end; j++)
        {
            out_data[j] = large_data[j];
        }
    }
    t2 = omp_get_wtime();
    BW = 2.0*(large_size/128)*sizeof(int) / (1e9 * (t2 - t1));
    cout << "one by one BW (no vect): " << BW << " GB/s " << 100.0*BW/1200 << " %" << endl;

    delete []large_data;
    delete []borders;
}
