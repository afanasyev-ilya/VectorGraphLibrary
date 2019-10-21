//
//  page_rank_verification.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 06/05/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef page_rank_verification_hpp
#define page_rank_verification_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void page_rank_verification_small_graphs(int _vertices_count = 16, int _avg_connections = 4)
{
    long long edges_count = _vertices_count * _avg_connections;
    EdgesListGraph<int, float> rand_graph;
    //GraphGenerationAPI<int, float>::random_uniform(rand_graph, _vertices_count, edges_count);
    GraphGenerationAPI<int, float>::R_MAT(rand_graph, _vertices_count, edges_count, 57, 19, 19, 5);
    //GraphGenerationAPI<int, float>::SSCA2(rand_graph, _vertices_count, _avg_connections);
    rand_graph.save_to_graphviz_file("edges_list_verification.gv");
    
    
    VectorisedCSRGraph<int, float> vect_graph;
    vect_graph.import_graph(rand_graph, VERTICES_SORTED, EDGES_SORTED, VECTOR_LENGTH, PULL_TRAVERSAL);
    vect_graph.print();
    
    float *page_ranks;
    
    PageRank<int, float>::allocate_result_memory(vect_graph.get_vertices_count(), &page_ranks);
    
    PageRank<int, float>::page_rank(vect_graph, page_ranks, 1);
    
    PageRank<int, float>::free_result_memory(page_ranks);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* page_rank_verification_hpp */
