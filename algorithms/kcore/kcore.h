//
//  kcore.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 01/10/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef kcore_h
#define kcore_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class KCore
{
private:
    
public:
    static void allocate_result_memory(int _vertices_count, int **_kcore_data);
    static void free_result_memory    (int *_kcore_data);
    
    static void calculate_kcore_sizes(UndirectedGraph &_graph,
                                      int *_kcore_data,
                                      int &_vertices_count,
                                      long long &_edges_count);
    
    // returns int *_kcore_degrees array of size |V|, i-th vertex in kcore <=> _kcore_degrees[i] > 0
    static void kcore_subgraph(UndirectedGraph &_graph, int *_kcore_degrees, int _k);
    static void kcore_subgraph(VectorisedCSRGraph &_graph, int *_kcore_degrees, int _k);
    
    static void maximal_kcore(UndirectedGraph &_graph, int *_kcore_degrees);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "kcore.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* kcore_h */
