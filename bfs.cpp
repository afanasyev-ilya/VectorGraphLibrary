//
//  bfs.cpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 12/09/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#include <stdio.h>
#include "graph_library.h"
#include <iostream>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        cout << "BFS (Breadth-First Search) test..." << endl;
        
        // parse args
        AlgorithmCommandOptionsParser parser;
        parser.parse_args(argc, argv);
        
        // load graph
        /*EdgesListGraph<int, float> rand_graph;
        int vertices_count = pow(2.0, parser.get_scale());
        long long edges_count = vertices_count * parser.get_avg_degree();
        GraphGenerationAPI<int, float>::random_uniform(rand_graph, vertices_count, edges_count, UNDIRECTED_GRAPH);*/
        
        ExtendedCSRGraph<int, float> ext_graph;
        ext_graph.load_from_binary_file(parser.get_graph_file_name());
        //ext_graph.import_graph(rand_graph, VERTICES_SORTED, EDGES_SORTED, 1, PULL_TRAVERSAL);
        
        // compute CC
        int *bfs_result = new int[ext_graph.get_vertices_count()];
        BFS<int, float>::new_bfs(ext_graph, bfs_result, 0);
        
        BFS<int, float>::verifier(ext_graph, 0, bfs_result);
        
        delete []bfs_result;
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
