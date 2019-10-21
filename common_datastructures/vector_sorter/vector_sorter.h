//
//  vector_sorter.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 22/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef vector_sorter_h
#define vector_sorter_h

#define ACCESS_SORT_DATA(i, j, dim_size) ((j) * (dim_size) + (i))

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class VectorSorter
{
private:
    int max_size, current_size;
    int dim_size;
    
    int *tmp_data;
    
    void initial_sort();
    void merge_sorted_arrays(int *_data, int _first_start, int *_result_data, int _second_start, int _size);
public:
    VectorSorter(int _size);
    ~VectorSorter();
    
    void sort_data(int *_data, int _current_size);
    void find_max(int *_most_frequent_labels, int *_current_labels);
    
    void clear();
    void print();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "vector_sorter.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* vector_sorter_h */
