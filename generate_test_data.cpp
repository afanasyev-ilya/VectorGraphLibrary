//
//  generate_test_data.cpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 16/04/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#include <stdio.h>
#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void parse_cmd_params(int _argc, char **_argv, int &_scale, int &_avg_degree, string &_graph_type,
                      string &_output_format, string &_file_name, bool &_convert, string &_input_file_name,
                      SupportedTraversalType &_traversal_type,
                      bool &_append_with_reverse_edges, EdgesState &_edges_state,
                      DirectionType &_direction_type)
{
    // set deafualt params
    _scale = 10;
    _avg_degree = 16;
    _graph_type = "RMAT";
    _output_format = "vectorised_CSR";
    _file_name = "rng_graph.gbin";
    _convert = false;
    _input_file_name = "wiki.txt";
    _traversal_type = PULL_TRAVERSAL;
    _append_with_reverse_edges = false;
    _edges_state = EDGES_SORTED;
    _direction_type = UNDIRECTED_GRAPH;
    
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
        
        if(option.compare("-append_with_reverse_edges") == 0)
        {
            _append_with_reverse_edges = true;
        }
        
        if(option.compare("-edges_random_shuffled") == 0)
        {
            _edges_state = EDGES_RANDOM_SHUFFLED;
        }
        
        if(option.compare("-edges_unsorted") == 0)
        {
            _edges_state = EDGES_UNSORTED;
        }
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void save_to_file(vector<int> vals, string file_name)
{
    ofstream myfile;
    myfile.open(file_name.c_str());
    
    for(int i = 0; i < vals.size(); i++)
        myfile << vals[i] << "\n";
    
    myfile.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*void generate_memory_profile(EdgesListGraph<int, float> &rand_graph)
{
    int vect_size = 1000;
    
    int *whole_array = new int[3*vect_size];
    int *a = &whole_array[0];
    int *b = &whole_array[vect_size];
    int *c = &whole_array[2*vect_size];
    
    vector<int> saxpy_accesses;
    
    for(int i = 0; i < vect_size; i++)
    {
        a[i] = b[i] + c[i];
        saxpy_accesses.push_back(&c[i] - whole_array);
        saxpy_accesses.push_back(&b[i] - whole_array);
        saxpy_accesses.push_back(&a[i] - whole_array);
    }
    
    save_to_file(saxpy_accesses, "saxpy.txt");
    
    int edges_count = vect_size;
    int vertices_count = vect_size;
    int *distances = &whole_array[0];
    int *dst_ids = &whole_array[vertices_count];
    for(int i = 0; i < edges_count; i++)
    {
        dst_ids[i] = rand() % vertices_count;
    }
    
    vector<int> random_accesses;
    for(int i = 0; i < edges_count; i++)
    {
        int dst_id = dst_ids[i];
        random_accesses.push_back(&dst_ids[i] - whole_array);
        
        int val = distances[dst_id];
        random_accesses.push_back(&distances[dst_id] - whole_array);
    }
    
    save_to_file(random_accesses, "rmat.txt");
    
    for(int i = 0; i < edges_count; i++)
    {
        dst_ids[i] = rand_graph.get_dst_ids();
    }
    
    vector<int> rmat_accesses;
    for(int i = 0; i < edges_count; i++)
    {
        int dst_id = dst_ids[i];
        random_accesses.push_back(&dst_ids[i] - whole_array);
        
        int val = distances[dst_id];
        random_accesses.push_back(&distances[dst_id] - whole_array);
    }
    
    save_to_file(random_accesses, "random.txt");
    
    delete []whole_array;
}*/

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
        SupportedTraversalType traversal_type;
        bool append_with_reverse_edges;
        DirectionType direction_type;
        EdgesState edges_state;
        
        parse_cmd_params(argc, argv, scale, avg_degree, graph_type, output_format, file_name, convert, input_file_name,
                         traversal_type, append_with_reverse_edges, edges_state, direction_type);
        
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
            cout << "Generate time: " << (t2 - t1) * 1000.0 << " ms" << endl;
        }
        
        double old_edges_count = rand_graph.get_edges_count();
        double edges_list_size = ((double)rand_graph.get_edges_count()) * (2.0*sizeof(int) + sizeof(float));
        double CSR_size = ((double)rand_graph.get_edges_count()) * (1.0*sizeof(int) + sizeof(float)) + ((double)rand_graph.get_vertices_count()) * (sizeof(long long int));
        cout << "original vertices count: " << rand_graph.get_vertices_count() << endl;
        cout << "original edges count: " << rand_graph.get_edges_count() << endl;
        cout << "estimated EDGES_LIST size: " << edges_list_size / 1e9 << " GB" << endl;
        cout << "estimated TRADITIONAL_CSR size: " << CSR_size / 1e9 << " GB" << endl << endl;
        
        if(output_format.find("edges_list") != string::npos)
        {
            rand_graph.save_to_binary_file(file_name + "_edges_list.gbin");
        }
        else if(output_format.find("extended_CSR") != string::npos)
        {
            ExtendedCSRGraph<int, float> result_graph;
            result_graph.import_graph(rand_graph, VERTICES_SORTED, edges_state, VECTOR_LENGTH, traversal_type);
            result_graph.save_to_binary_file(file_name + "_ext_CSR.gbin");
            cout << "saved into ExtendedCSRGraph!" << endl;
        }
        else if(output_format.find("vectorised_CSR") != string::npos)
        {
            VectorisedCSRGraph<int, float> result_graph;
            result_graph.import_graph(rand_graph, VERTICES_SORTED, edges_state, VECTOR_LENGTH, traversal_type, true);
            result_graph.save_to_binary_file(file_name + "_vect_CSR.gbin");
            double new_edges_count = result_graph.get_edges_count();
            cout << "saved into VectorisedCSRGraph!" << endl;
            cout << "edges count in VectorisedCSRGraph: " << new_edges_count << endl;
            cout << "added " << (new_edges_count - old_edges_count) * 100.0 / old_edges_count << " % extra edges" << endl << endl;
        }
        else if(output_format.find("sharded") != string::npos)
        {
            ShardedGraph<int, float> sharded_graph(SHARD_VECT_CSR_TYPE, LLC_CACHE_SIZE);
            sharded_graph.import_graph(rand_graph, traversal_type); // need to set edges_direction here
            sharded_graph.print_stats();
            
            sharded_graph.save_to_binary_file(file_name + "_sharded.gbin");
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
            export_to_gapbs_text_unweighted(rand_graph, file_name + "_gapbs.el");
            t1 = omp_get_wtime();
            cout << "save time: " << (t2 - t1) * 1000.0 << " ms" << endl;
            cout << "saved into GAPBS format!" << endl;
        }
        
        //generate_memory_profile();
    }
    catch (const char * error)
    {
        cout << error << endl;
    }
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
