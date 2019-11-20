//
//  custom_test.cpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 04/10/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"
#include <iostream>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        double t1,t2;
        cout << "Custom test..." << endl;
        
        int size = pow(2.0, 23);
        
        int *in_data = new int[size];
        int *out_data = new int[size];
        int *tmp_buffer = new int[size];
        
        int val_to_find = 1;
        
        #pragma omp parallel for
        for(int i = 0; i < size; i++)
        {
            in_data[i] = 0;
        }
        
        vector<double> sparse_bandwidths;
        vector<double> dense_bandwidths;
        vector<string> numbers_of_elements;
        
        int divider = 1024*1024;
        while(divider >= 1)
        {
            int number_of_active_elements = size / divider;
            cout << "number of elements for test: " << number_of_active_elements << endl;
            cout << "percent: " << (100.0 * number_of_active_elements)/size << " %" << endl;
            
            for(int i = 0; i < number_of_active_elements; i++)
            {
                int dst_id = rand() % size;
                in_data[dst_id] = val_to_find;
            }
            
            cout << "SPARSE" << endl;
            t1 = omp_get_wtime();
            sparse_copy_if(in_data, out_data, tmp_buffer, size, val_to_find);
            t2 = omp_get_wtime();
            cout << "time: " << (t2 - t1) * 1000.0 << " ms" << endl;
            cout << "bandwidth: " << 2.0 * size * sizeof(int) / ((t2 - t1)*1e9) << " GB/s" << endl;
            
            sparse_bandwidths.push_back(2.0 * size * sizeof(int) / ((t2 - t1)*1e9));
            
            cout << "DENSE" << endl;
            t1 = omp_get_wtime();
            dense_copy_if(in_data, out_data, size, val_to_find);
            t2 = omp_get_wtime();
            cout << "time: " << (t2 - t1) * 1000.0 << " ms" << endl;
            cout << "bandwidth: " << 2.0 * size * sizeof(int) / ((t2 - t1)*1e9) << " GB/s" << endl << endl;
            
            dense_bandwidths.push_back(2.0 * size * sizeof(int) / ((t2 - t1)*1e9));
            
            string info = std::to_string(number_of_active_elements) + string(" (") + std::to_string((100.0 * number_of_active_elements)/size) + string("%)");
            numbers_of_elements.push_back(info);
            
            divider /= 2;
        }
        
        for(int i = 0; i < sparse_bandwidths.size(); i++)
            cout << sparse_bandwidths[i] << endl;
        cout << endl << endl;
        
        for(int i = 0; i < dense_bandwidths.size(); i++)
            cout << dense_bandwidths[i] << endl;
        cout << endl << endl;
        
        for(int i = 0; i < numbers_of_elements.size(); i++)
            cout << numbers_of_elements[i] << endl;
        cout << endl << endl;
        
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

