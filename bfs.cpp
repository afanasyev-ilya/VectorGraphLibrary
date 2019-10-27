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
        ExtendedCSRGraph<int, float> graph;
        EdgesListGraph<int, float> rand_graph;
        if(parser.get_compute_mode() == GENERATE_NEW_GRAPH)
        {
            int vertices_count = pow(2.0, parser.get_scale());
            long long edges_count = vertices_count * parser.get_avg_degree();
            //GraphGenerationAPI<int, float>::random_uniform(rand_graph, vertices_count, edges_count, UNDIRECTED_GRAPH);
            GraphGenerationAPI<int, float>::R_MAT(rand_graph, vertices_count, edges_count, 57, 19, 19, 5, UNDIRECTED_GRAPH);
            graph.import_graph(rand_graph, VERTICES_SORTED, EDGES_SORTED, 1, PUSH_TRAVERSAL);
        }
        else if(parser.get_compute_mode() == LOAD_GRAPH_FROM_FILE)
        {
            if(!graph.load_from_binary_file(parser.get_graph_file_name()))
                throw "ERROR: graph file not found";
        }

        // compute CC
        int *bfs_result = new int[graph.get_vertices_count()];
        BFS<int, float>::nec_direction_optimising_BFS(graph, bfs_result, 1);
        cout << endl << "-------------------------------------------------" << endl << endl;
        BFS<int, float>::nec_direction_optimising_BFS(graph, bfs_result, 10);
        cout << endl << "-------------------------------------------------" << endl << endl;
        /*BFS<int, float>::nec_direction_optimising_BFS(graph, bfs_result, 100);
        cout << endl << "-------------------------------------------------" << endl << endl;
        BFS<int, float>::nec_direction_optimising_BFS(graph, bfs_result, 20);
        cout << endl << "-------------------------------------------------" << endl << endl;*/
        
        BFS<int, float>::verifier(graph, 10, bfs_result);
        
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
