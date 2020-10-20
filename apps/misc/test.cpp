/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define __USE_INTEL__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        int size = 64*1024*1024;

        float *a, *b, *c;
        MemoryAPI::allocate_array(&a, size);
        MemoryAPI::allocate_array(&c, size);
        MemoryAPI::allocate_array(&b, size);

        int large_size = 64*1024*1024;
        int cached_size = 2*4096;
        int *dst_ids;
        MemoryAPI::allocate_array(&dst_ids, large_size);
        float *result;
        MemoryAPI::allocate_array(&result, large_size);

        for(int i = 0; i < large_size; i++)
        {
            dst_ids[i] = rand()%cached_size;
        }

        for(int i = 0; i < size; i++)
        {
            a[i] = rand()%100;
            b[i] = rand()%100;
        }

        for(int tr = 0; tr < 5; tr++)
        {
            Timer tm;
            tm.start();
            #pragma omp parallel for simd
            for(int i = 0; i < size; i++)
            {
                c[i] = a[i] + b[i];
            }
            tm.end();
            tm.print_bandwidth_stats("KNL", size, 3.0*sizeof(float));

            tm.start();
            #pragma omp parallel for simd
            for(int i = 0; i < large_size; i++)
            {
                result[i] = a[dst_ids[i]];
            }
            tm.end();
            tm.print_bandwidth_stats("KNL gather", large_size, 3.0*sizeof(float));

        }
        cout << " --------------------------------------------- " << endl;

        MemoryAPI::free_array(a);
        MemoryAPI::free_array(c);
        MemoryAPI::free_array(b);
        MemoryAPI::free_array(result);
        MemoryAPI::free_array(dst_ids);
    }
    catch (string error)
    {
        cout << error << endl;
    }
    catch (const char * error)
    {
        cout << error << endl;
    }
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
