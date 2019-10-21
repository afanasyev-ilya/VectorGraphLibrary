//
//  cc.cpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 07/09/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#include "graph_library.h"
#include <iostream>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        cout << "CC (connected components) test..." << endl;
        
        // parse args
        AlgorithmCommandOptionsParser parser;
        parser.parse_args(argc, argv);
        
        // load graph
        VectorisedCSRGraph<int, float> graph;
        EdgesListGraph<int, float> rand_graph;
        if(parser.get_compute_mode() == GENERATE_NEW_GRAPH)
        {
            int vertices_count = pow(2.0, parser.get_scale());
            long long edges_count = vertices_count * parser.get_avg_degree();
            //GraphGenerationAPI<int, float>::random_uniform(rand_graph, vertices_count, edges_count, UNDIRECTED_GRAPH);
            GraphGenerationAPI<int, float>::R_MAT(rand_graph, vertices_count, edges_count, 57, 19, 19, 5, UNDIRECTED_GRAPH);
            graph.import_graph(rand_graph, VERTICES_SORTED, EDGES_SORTED, VECTOR_LENGTH, PULL_TRAVERSAL);
        }
        else if(parser.get_compute_mode() == LOAD_GRAPH_FROM_FILE)
        {
            if(!graph.load_from_binary_file(parser.get_graph_file_name()))
                throw "ERROR: graph file not found";
        }
        
        // compute CC
        cout << "Computations started..." << endl;
        int *cc_result;
        ConnectedComponents<int, float>::allocate_result_memory(graph.get_vertices_count(), &cc_result);
        ConnectedComponents<int, float>::nec_shiloach_vishkin(graph, cc_result);
        
        // check if required
        if(parser.get_check_flag() && (parser.get_compute_mode() == GENERATE_NEW_GRAPH))
        {
            ExtendedCSRGraph<int, float> ext_graph;
            ext_graph.import_graph(rand_graph, VERTICES_SORTED, EDGES_SORTED, 1, PULL_TRAVERSAL);
            
            int *ext_cc_result;
            ConnectedComponents<int, float>::allocate_result_memory(ext_graph.get_vertices_count(), &ext_cc_result);
            ConnectedComponents<int, float>::nec_shiloach_vishkin(ext_graph, ext_cc_result);
            
            verify_results(cc_result, ext_cc_result, min(graph.get_vertices_count(), ext_graph.get_vertices_count()));
            
            ConnectedComponents<int, float>::free_result_memory(ext_cc_result);
        }
        
        ConnectedComponents<int, float>::free_result_memory(cc_result);
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

