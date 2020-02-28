//
//  generate_test_data.cpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 16/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#include <stdio.h>
#include "../graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void parse_cmd_params(int _argc, char **_argv, int &_scale, int &_avg_degree, string &_graph_type,
                      string &_output_format, string &_file_name, bool &_convert, string &_input_file_name,
                      TraversalDirection &_traversal_type,
                      bool &_append_with_reverse_edges, EdgesState &_edges_state,
                      DirectionType &_direction_type, MultipleArcsState &_multiple_arcs_state)
{
    // set deafualt params
    _scale = 10;
    _avg_degree = 16;
    _graph_type = "RMAT";
    _output_format = "extended_CSR";
    _file_name = "rng_graph.gbin";
    _convert = false;
    _input_file_name = "wiki.txt";
    _traversal_type = PULL_TRAVERSAL;
    _append_with_reverse_edges = false;
    _edges_state = EDGES_SORTED;
    _direction_type = UNDIRECTED_GRAPH;
    _multiple_arcs_state = MULTIPLE_ARCS_PRESENT;
    
    // get params from cmd line
    for (int i = 1; i < _argc; i++)
    {
        string option(_argv[i]);
        
        if (option.compare("-s") == 0)
        {
            _scale = atoi(_argv[++i]);
        }
        
        if (option.compare("-e") == 0)
        {
            _avg_degree = atoi(_argv[++i]);
        }
        
        if (option.compare("-type") == 0)
        {
            _graph_type = _argv[++i];
        }
        
        if (option.compare("-format") == 0)
        {
            _output_format = _argv[++i];
        }
        
        if (option.compare("-file") == 0)
        {
            _file_name = _argv[++i];
        }
        
        if (option.compare("-convert") == 0)
        {
            _convert = true;
            _input_file_name = _argv[++i];
        }
        
        if (option.compare("-pull") == 0)
        {
            _traversal_type = PULL_TRAVERSAL;
        }
        
        if (option.compare("-push") == 0)
        {
            _traversal_type = PUSH_TRAVERSAL;
        }
        
        if (option.compare("-directed") == 0)
        {
            _direction_type = DIRECTED_GRAPH;
        }
        
        if(option.compare("-append-with-reverse-edges") == 0)
        {
            _append_with_reverse_edges = true;
        }
        
        if(option.compare("-edges-random-shuffled") == 0)
        {
            _edges_state = EDGES_RANDOM_SHUFFLED;
        }
        
        if(option.compare("-edges-unsorted") == 0)
        {
            _edges_state = EDGES_UNSORTED;
        }

        if(option.compare("-multiple-arcs-removed") == 0)
        {
            _multiple_arcs_state = MULTIPLE_ARCS_REMOVED;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char ** argv)
{
    try
    {
        double t1, t2;
        
        int scale, avg_degree;
        string graph_type, output_format, file_name;
        bool convert;
        string input_file_name;
        TraversalDirection traversal_type;
        bool append_with_reverse_edges;
        DirectionType direction_type;
        EdgesState edges_state;
        MultipleArcsState multiple_arcs_state;
        
        parse_cmd_params(argc, argv, scale, avg_degree, graph_type, output_format, file_name, convert, input_file_name,
                         traversal_type, append_with_reverse_edges, edges_state, direction_type, multiple_arcs_state);
        
        EdgesListGraph<int, float> rand_graph;
        if(convert)
        {
            GraphGenerationAPI<int, float>::init_from_txt_file(rand_graph, input_file_name, append_with_reverse_edges);
        }
        else
        {
            t1 = omp_get_wtime();
            int vertices_count = pow(2.0, scale);
            long long edges_count = (long long)vertices_count * (long long)avg_degree;
            if(graph_type == "RMAT" || graph_type == "rmat")
            {
                cout << "Generating RMAT with: " << vertices_count << " vertices and " << edges_count << " edges" << endl;
                GraphGenerationAPI<int, float>::R_MAT(rand_graph, vertices_count, edges_count, 57, 19, 19, 5, direction_type);
            }
            else if(graph_type == "random_uniform" || graph_type == "ru" || graph_type == "random-uniform")
            {
                cout << "Generating random_uniform with: " << vertices_count << " vertices and " << edges_count << " edges" << endl;
                GraphGenerationAPI<int, float>::random_uniform(rand_graph, vertices_count, edges_count, direction_type);
            }
            else if(graph_type == "SSCA2" || graph_type == "ssca2")
            {
                cout << "Generating SSCA2 with: " << vertices_count << " vertices and " << edges_count << " edges" << endl;
                GraphGenerationAPI<int, float>::SSCA2(rand_graph, vertices_count, avg_degree);
            }
            else
            {
                cout << "Unknown graph type" << endl;
            }
            t2 = omp_get_wtime();
            cout << "Generate time: " << t2 - t1 << " sec" << endl;
        }
        
        double old_edges_count = rand_graph.get_edges_count();
        double edges_list_size = ((double)rand_graph.get_edges_count()) * (2.0*sizeof(int) + sizeof(float));
        double CSR_size = ((double)rand_graph.get_edges_count()) * (1.0*sizeof(int) + sizeof(float)) + ((double)rand_graph.get_vertices_count()) * (sizeof(long long int));
        cout << "original vertices count: " << rand_graph.get_vertices_count() << endl;
        cout << "original edges count: " << rand_graph.get_edges_count() << endl;
        cout << "estimated EDGES_LIST size: " << edges_list_size / 1e9 << " GB" << endl;
        cout << "estimated TRADITIONAL_CSR size: " << CSR_size / 1e9 << " GB" << endl << endl;
        
        if(output_format.find("extended_CSR") != string::npos)
        {
            ExtendedCSRGraph<int, float> result_graph;
            t1 = omp_get_wtime();
            result_graph.import_graph(rand_graph, VERTICES_SORTED, edges_state, 1, traversal_type, multiple_arcs_state);
            t2 = omp_get_wtime();
            cout << "format conversion time: " << t2 - t1 << " sec" << endl;
            
            t1 = omp_get_wtime();
            result_graph.save_to_binary_file(file_name + "_ext_CSR.gbin");
            t2 = omp_get_wtime();
            cout << "saved into ExtendedCSRGraph in " << t2 - t1 << " sec"  << endl;
        }
        else if(output_format.find("ligra") != string::npos)
        {
            t1 = omp_get_wtime();
            export_to_ligra_text_unweighted(rand_graph, file_name + "_ligra.txt");
            t2 = omp_get_wtime();
            cout << "save time: " << (t2 - t1) * 1000.0 << " ms" << endl;
            cout << "saved into Ligra format!" << endl;
        }
        else if(output_format.find("gapbs") != string::npos)
        {
            t1 = omp_get_wtime();
            export_to_edges_list_unweighted(rand_graph, file_name + "_gapbs.el", false);
            t1 = omp_get_wtime();
            cout << "save time: " << (t2 - t1) * 1000.0 << " ms" << endl;
            cout << "saved into GAPBS format!" << endl;
        }
        else if(output_format.find("mtx") != string::npos)
        {
            t1 = omp_get_wtime();
            export_to_edges_list_unweighted(rand_graph, file_name + "_mtx.el", true);
            t1 = omp_get_wtime();
            cout << "save time: " << (t2 - t1) * 1000.0 << " ms" << endl;
            cout << "saved into MTX format!" << endl;
        }
    }
    catch (const char * error)
    {
        cout << error << endl;
    }
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
