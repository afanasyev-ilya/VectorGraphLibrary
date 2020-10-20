/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define __USE_INTEL__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        int size = 16*1024*1024;

        float *a, *b, *c;
        MemoryAPI::allocate_array(&a, size);
        MemoryAPI::allocate_array(&c, size);
        MemoryAPI::allocate_array(&b, size);

        for(int i = 0; i < size; i++)
        {
            a[i] = rand()%100;
            b[i] = rand()%100;
        }

        for(int tr = 0; tr < 5; tr++)
        {
            Timer tm;
            tm.start();
            #pragma simd
            #pragma omp parallel for schedule(static, 2048)
            for(int i = 0; i < size; i++)
            {
                c[i] = a[i] + b[i];
            }
            tm.end();
            float sum = 0;
            for(int i = 0; i < size; i++)
            {
                sum += c[i];
            }
            cout << " sum " << sum << endl;
            tm.print_bandwidth_stats("KNL", size, 3.0*sizeof(float));

        }
        cout << " --------------------------------------------- " << endl;

        MemoryAPI::free_array(a);
        MemoryAPI::free_array(c);
        MemoryAPI::free_array(b);
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
