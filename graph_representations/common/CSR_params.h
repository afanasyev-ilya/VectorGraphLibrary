//
//  CSR_params.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 19/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef CSR_params_h
#define CSR_params_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum VerticesState
{
    VERTICES_SORTED = 1,
    VERTICES_UNSORTED = 0,
    VERTICES_RANDOM_SHUFFLED = 2
};

enum EdgesState
{
    EDGES_SORTED = 1,
    EDGES_UNSORTED = 0,
    EDGES_RANDOM_SHUFFLED = 2
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* CSR_params_h */
