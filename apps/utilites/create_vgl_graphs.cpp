/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define __USE_INTEL__

#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void parse_cmd_params(int _argc, char **_argv, int &_scale, int &_avg_degree, string &_graph_type,
                      string &_output_format, string &_file_name, bool &_convert, string &_input_file_name,
                      bool &_append_with_reverse_edges, EdgesState &_edges_state,
                      MultipleArcsState &_multiple_arcs_state)
{
    // set deafualt params
    _scale = 10;
    _avg_degree = 16;
    _graph_type = "RMAT";
    _output_format = "vect_csr";
    _file_name = "test.vgraph";
    _convert = false;
    _input_file_name = "wiki.txt";
    _append_with_reverse_edges = false;
    _edges_state = EDGES_SORTED;
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

        if(option.compare("-multiple-arcs-removed") == 0 || option.compare("-no-multiple-arcs") == 0)
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
        int scale, avg_degree;
        string graph_type, output_format, file_name;
        bool convert;
        string input_file_name;
        bool append_with_reverse_edges;
        EdgesState edges_state;
        MultipleArcsState multiple_arcs_state;
        
        parse_cmd_params(argc, argv, scale, avg_degree, graph_type, output_format, file_name, convert, input_file_name,
                         append_with_reverse_edges, edges_state, multiple_arcs_state);

        EdgesListGraph rand_graph;
        if(convert)
        {
            cout << " --------------------------- " << "converting " << file_name << "--------------------------- " << endl;
            GraphGenerationAPI::init_from_txt_file(rand_graph, input_file_name, append_with_reverse_edges);
        }
        else
        {
            cout << " --------------------------- " << "generating " << file_name << "--------------------------- " << endl;
            Timer tm;
            tm.start();
            int vertices_count = pow(2.0, scale);
            long long edges_count = (long long)vertices_count * (long long)avg_degree;
            if(graph_type == "RMAT" || graph_type == "rmat")
            {
                cout << "Generating RMAT with: " << vertices_count << " vertices and " << edges_count << " edges" << endl;
                GraphGenerationAPI::R_MAT(rand_graph, vertices_count, edges_count, 57, 19, 19, 5, DIRECTED_GRAPH);
            }
            else if(graph_type == "random_uniform" || graph_type == "ru" || graph_type == "random-uniform")
            {
                cout << "Generating random_uniform with: " << vertices_count << " vertices and " << edges_count << " edges" << endl;
                GraphGenerationAPI::random_uniform(rand_graph, vertices_count, edges_count, DIRECTED_GRAPH);
            }
            else
            {
                cout << "Unknown graph type" << endl;
            }
            tm.end();
            tm.print_time_stats("Generate");
        }

        if((output_format.find("vect_csr") != string::npos) || (output_format.find("vect_CSR") != string::npos))
        {
            VectCSRGraph vect_csr_graph;
            Timer tm;
            tm.start();
            vect_csr_graph.import(rand_graph);
            tm.end();
            tm.print_time_stats("Import");

            tm.start();
            add_extension(file_name, ".vgraph");
            vect_csr_graph.save_to_binary_file(file_name);
            cout << "VectCSR graph is generated and saved to file " << file_name << endl;
            tm.end();
            tm.print_time_stats("Save");
        }
        cout << " ----------------------------------------------------------------------------------------- " << endl << endl;
    }
    catch (const char * error)
    {
        cout << error << endl;
    }
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////