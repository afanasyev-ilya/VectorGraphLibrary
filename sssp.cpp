//
//  sssp.cpp
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
        cout << "SSSP (Single Source Shortest Paths) test..." << endl;
        
        // parse args
        AlgorithmCommandOptionsParser parser;
        parser.parse_args(argc, argv);
        
        // load graph
        double t1 = omp_get_wtime();
        //VectorisedCSRGraph<int, float> graph;
        ExtendedCSRGraph<int, float> graph;
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
        double t2 = omp_get_wtime();
        cout << "Load graph time: " << t2 - t1 << " sec" << endl;
        
        // compute CC
        int last_src_vertex = 0;
        cout << "Computations started..." << endl;
        float *distances;
        ShortestPaths<int, float>::allocate_result_memory(graph.get_vertices_count(), &distances);
        
        #ifdef __USE_GPU__
        graph.move_to_device();
        #endif

        cout << "Doing " << parser.get_steps_count() << " SSSP iterations" << endl;
        t1 = omp_get_wtime();
        for(int i = 0; i < parser.get_steps_count(); i++)
        {
            last_src_vertex = i; //rand() % (graph.get_vertices_count()/4);
            #ifdef __USE_NEC_SX_AURORA__
            //ShortestPaths<int, float>::nec_dijkstra(graph, last_src_vertex, distances);
            ShortestPaths<int, float>::lib_dijkstra(graph, last_src_vertex, distances);
            #endif
            
            #ifdef __USE_GPU__
            ShortestPaths<int, float>::gpu_dijkstra(graph, last_src_vertex, distances);
            #endif
        }
        t2 = omp_get_wtime();
        
        #ifdef __USE_GPU__
        graph.move_to_host();
        #endif
        
        cout << "SSSP wall time: " << t2 - t1 << " sec" << endl;
        cout << "SSSP average performance: " << parser.get_steps_count() * (((double)graph.get_edges_count()) / ((t2 - t1) * 1e6)) << " MFLOPS" << endl << endl;
        
        // check if required
        if(parser.get_check_flag())
        {
            float *ext_distances;
            ShortestPaths<int, float>::allocate_result_memory(graph.get_vertices_count(), &ext_distances);
            ShortestPaths<int, float>::seq_dijkstra(graph, last_src_vertex, ext_distances);
            
            verify_results(distances, ext_distances, graph.get_vertices_count());
            
            ShortestPaths<int, float>::free_result_memory(ext_distances);
        }
        
        ShortestPaths<int, float>::free_result_memory(distances);
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
