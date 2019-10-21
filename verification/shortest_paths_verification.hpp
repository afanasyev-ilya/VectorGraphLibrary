//
//  shortest_paths_verification.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 25/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef shortest_paths_verification_h
#define shortest_paths_verification_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void shortest_paths_verification_small_graphs(int _vertices_count = 16, int _avg_connections = 4)
{
    long long edges_count = _vertices_count * _avg_connections;
    EdgesListGraph<int, float> rand_graph;
    //GraphGenerationAPI<int, float>::random_uniform(rand_graph, _vertices_count, edges_count);
    GraphGenerationAPI<int, float>::R_MAT(rand_graph, _vertices_count, edges_count, 57, 19, 19, 5);
    //GraphGenerationAPI<int, float>::SSCA2(rand_graph, _vertices_count, _avg_connections);
    //rand_graph.save_to_graphviz_file("edges_list_verification.gv");
    
    float *ideal_result;
    ShortestPaths<int, float>::allocate_result_memory(rand_graph.get_vertices_count(), &ideal_result);
    ShortestPaths<int, float>::bellman_ford(rand_graph, 0, ideal_result);
    
    VectorisedCSRGraph<int, float> vect_graph;
    vect_graph.import_graph(rand_graph, VERTICES_SORTED, EDGES_SORTED, VECTOR_LENGTH, PULL_TRAVERSAL);
    vect_graph.save_to_binary_file("test.gbin");
    vect_graph.load_from_binary_file("test.gbin");
    //vect_graph.print();
    
    float *first_result;
    ShortestPaths<int, float>::allocate_result_memory(vect_graph.get_vertices_count(), &first_result);
    ShortestPaths<int, float>::bellman_ford(vect_graph, vect_graph.get_reordered_vertex_ids()[0], first_result);
    ShortestPaths<int, float>::reorder_result(vect_graph, first_result);
    
    int error_count = 0;
    for(int i = 0; i < vect_graph.get_vertices_count(); i++)
    {
        if(fabs(ideal_result[i] - first_result[i]) > 0.1)
        {
            //if(error_count < 20)
            cout << "Error: " << ideal_result[i] << " vs " << first_result[i] << " in pos " << i << endl;
            error_count++;
        }
        //cout << "Error: " << ideal_result[i] << " | " << first_result[i] << " vs " << second_result[i] << " in pos " << i << endl;
    }
    cout << "error count: " << error_count << endl;
    
    ShortestPaths<int, float>::free_result_memory(first_result);
    ShortestPaths<int, float>::free_result_memory(ideal_result);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* shortest_paths_verification_h */
