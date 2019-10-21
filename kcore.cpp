//
//  kcore.cpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 01/10/2019.
//  Copyright © 2019 MSU. All rights reserved.

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"
#include <iostream>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        cout << "KCore test..." << endl;
        
        // parse args
        AlgorithmCommandOptionsParser parser;
        parser.parse_args(argc, argv);
        
        // gen new graph
        EdgesListGraph<int, float> rand_graph;
        int vertices_count = pow(2.0, atoi(argv[1]));
        int edges_count = vertices_count * atoi(argv[2]);
        GraphGenerationAPI<int, float>::R_MAT(rand_graph, vertices_count, edges_count, 57, 19, 19, 5, UNDIRECTED_GRAPH);
        
        ExtendedCSRGraph<int, float> graph;
        graph.import_graph(rand_graph, VERTICES_SORTED, EDGES_UNSORTED, VECTOR_LENGTH, PUSH_TRAVERSAL);
        
        //graph.save_to_graphviz_file("TEST.gv", VISUALISE_AS_UNDIRECTED);
        
        int *kcore_data;
        int vertices_in_kcore = 0;
        long long edges_in_kcore = 0;
        KCore<int, float>::allocate_result_memory(graph.get_vertices_count(), &kcore_data);
        KCore<int, float>::kcore_subgraph(graph, kcore_data, atoi(argv[3]));
        KCore<int, float>::calculate_kcore_sizes(graph, kcore_data, vertices_in_kcore, edges_in_kcore);
        cout << "vertices in kcore: " << vertices_in_kcore << "/" << graph.get_vertices_count() << endl;
        cout << "edges in kcore: " << edges_in_kcore << "/" << graph.get_edges_count() << endl;
        
        //KCore<int, float>::maximal_kcore(graph, kcore_data);
        
        VectorisedCSRGraph<int, float> vect_graph;
        vect_graph.import_graph(rand_graph, VERTICES_SORTED, EDGES_UNSORTED, VECTOR_LENGTH, PUSH_TRAVERSAL);
        KCore<int, float>::kcore_subgraph(vect_graph, kcore_data, atoi(argv[3]));
        
        KCore<int, float>::free_result_memory(kcore_data);
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
