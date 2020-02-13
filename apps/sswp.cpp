//
//  sswp.cpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 08/09/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#include "../graph_library.h"
#include <iostream>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[])
{
    try
    {
        cout << "SSWP (Single Source Widest Paths) test..." << endl;
        
        // parse args
        AlgorithmCommandOptionsParser parser;
        parser.parse_args(argc, argv);
        
        // load graph
        VectorisedCSRGraph<float, float> graph;
        EdgesListGraph<float, float> rand_graph;
        if(parser.get_compute_mode() == GENERATE_NEW_GRAPH)
        {
            int vertices_count = pow(2.0, parser.get_scale());
            long long edges_count = vertices_count * parser.get_avg_degree();
            //GraphGenerationAPI<float, float>::random_uniform(rand_graph, vertices_count, edges_count, UNDIRECTED_GRAPH);
            GraphGenerationAPI<float, float>::R_MAT(rand_graph, vertices_count, edges_count, 57, 19, 19, 5, DIRECTED_GRAPH);
            graph.import_graph(rand_graph, VERTICES_SORTED, EDGES_SORTED, VECTOR_LENGTH, PULL_TRAVERSAL);
        }
        else if(parser.get_compute_mode() == LOAD_GRAPH_FROM_FILE)
        {
            if(!graph.load_from_binary_file(parser.get_graph_file_name()))
            throw "ERROR: graph file not found";
        }
        
        // compute CC
        cout << "Computations started..." << endl;
        float *widths;
        WidestPaths<float, float>::allocate_result_memory(graph.get_vertices_count(), &widths);
        WidestPaths<float, float>::bellman_ford(graph, 0, widths);
        WidestPaths<float, float>::free_result_memory(widths);
        
        // check if required
        if(parser.get_check_flag() && (parser.get_compute_mode() == GENERATE_NEW_GRAPH))
        {
            ExtendedCSRGraph<float, float> ext_graph;
            ext_graph.import_graph(rand_graph, VERTICES_SORTED, EDGES_SORTED, 1, PULL_TRAVERSAL);

            float *ext_widths;
            WidestPaths<float, float>::allocate_result_memory(ext_graph.get_vertices_count(), &ext_widths);
            WidestPaths<float, float>::bellman_ford(ext_graph, 0, ext_widths);
            
            verify_results(widths, ext_widths, min(graph.get_vertices_count(), ext_graph.get_vertices_count()));
            
            WidestPaths<float, float>::free_result_memory(ext_widths);
        }
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
