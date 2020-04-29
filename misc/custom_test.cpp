//
//  custom_test.cpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 04/10/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define INT_ELEMENTS_PER_EDGE 1.0

#include "../graph_library.h"
#include <iostream>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        double t1,t2;
        cout << "Custom test..." << endl;
        
        int size = 4000000;
        
        int *in_data = new int[size];
        int *out_data = new int[size];
        int *tmp_buffer = new int[size];
        
        int val_to_find = 1;
        
        #pragma omp parallel for
        for(int i = 0; i < size; i++)
        {
            in_data[i] = rand() % 3;
            out_data[i] = 0;
            tmp_buffer[i] = 0;
        }

        t1 = omp_get_wtime();
        #pragma omp parallel for
        for(int vec_start = 0; vec_start < size; vec_start += VECTOR_LENGTH)
        {
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = vec_start + i;
                tmp_buffer[src_id] = in_data[src_id];
            }
        }
        t2 = omp_get_wtime();
        cout << "BW: " << 2.0 * sizeof(int) * size / ( (t2 - t1) * 1e9 ) << " GB/s" << endl;

        t1 = omp_get_wtime();
        #pragma omp parallel
        {
            int reg_data[VECTOR_LENGTH];
            int shifted_data[VECTOR_LENGTH];

            #pragma _NEC vreg(reg_data)
            #pragma _NEC vreg(shifted_data)

            #pragma _NEC vector
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                reg_data[i] = 0;
                shifted_data[i] = 0;
            }

            #pragma omp for schedule(static)
            for (int vec_start = 0; vec_start < size; vec_start += VECTOR_LENGTH)
            {
                #pragma _NEC vector
                for (int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int src_id = vec_start + i;
                    reg_data[i] = in_data[src_id];
                    shifted_data[i] = 0;
                }

                #pragma _NEC vector
                for (int i = 0; i < (VECTOR_LENGTH - 1); i++)
                {
                    shifted_data[i + 1] = reg_data[i];
                }

                #pragma _NEC vector
                for (int i = 0; i < VECTOR_LENGTH; i++)
                {
                    int src_id = vec_start + i;
                    out_data[src_id] = reg_data[i] + shifted_data[i];
                }
            }
        }
        t2 = omp_get_wtime();
        cout << "BW 2: " << 2.0 * sizeof(int) * size / ( (t2 - t1) * 1e9 ) << " GB/s" << endl;

        for (int i = 0; i < 16; i++)
        {
            cout << in_data[i] << " ";
        }
        for (int i = 0; i < 16; i++)
        {
            cout << out_data[i] << " ";
        }
        
        delete []in_data;
        delete []out_data;
        delete []tmp_buffer;
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

