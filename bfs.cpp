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
    

        BFS<int, float> bfs_operation;
        
        // compute CC
        int *bfs_result;
        bfs_operation.allocate_result_memory(graph.get_vertices_count(), &bfs_result);
        bfs_operation.init_temporary_datastructures(graph);
        
        vector<int> source_vertices = {1};
        int vertex_to_check = 0;
        
        cout << "Graph size: " << graph.get_edges_count()/graph.get_vertices_count() << endl;
        double avg_perf = 0;
        for(int i = 0; i < source_vertices.size(); i++)
        {
            vertex_to_check = source_vertices[i];
            //cout << "launching BFS from vertex: " << vertex_to_check << endl;
            
            double t1 = omp_get_wtime();
            bfs_operation.nec_direction_optimising_BFS(graph, bfs_result, vertex_to_check);
            double t2 = omp_get_wtime();
            
            //cout << "OUTER BFS Perf: " << ((double)graph.get_edges_count())/((t2-t1)*1e6) << " MTEPS" << endl;
            avg_perf += ((double)graph.get_edges_count())/((t2-t1)*1e6) / source_vertices.size();
            //cout << endl << "-------------------------------------------------" << endl << endl;
        }
        cout << "AVG Performance: " << avg_perf << " MTEPS" << endl << endl;
        
        bfs_operation.verifier(graph, vertex_to_check, bfs_result);
        bfs_operation.free_result_memory(bfs_result);
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
