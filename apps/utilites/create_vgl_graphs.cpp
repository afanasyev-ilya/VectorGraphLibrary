/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void parse_cmd_params(int _argc, char **_argv, int &_scale, int &_avg_degree, string &_graph_type,
                      string &_output_format, string &_file_name, bool &_convert, string &_input_file_name,
                      DirectionType &_direction_type)
{
    // set deafualt params
    _scale = 10;
    _avg_degree = 16;
    _graph_type = "RMAT";
    _output_format = "vect_csr";
    _file_name = "test.vgraph";
    _convert = false;
    _input_file_name = "wiki.txt";
    _direction_type = DIRECTED_GRAPH;
    
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
        
        if(option.compare("-directed") == 0)
        {
            _direction_type = DIRECTED_GRAPH;
        }

        if(option.compare("-undirected") == 0)
        {
            cout << "undirected is selected!" << endl;
            _direction_type = UNDIRECTED_GRAPH;
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
        DirectionType direction_type;
        
        parse_cmd_params(argc, argv, scale, avg_degree, graph_type, output_format, file_name, convert, input_file_name,
                         direction_type);

        EdgesListGraph rand_graph;
        if(convert)
        {
            cout << " --------------------------- " << "converting " << file_name << "--------------------------- " << endl;
            GraphGenerationAPI::init_from_txt_file(rand_graph, input_file_name, direction_type);
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
                GraphGenerationAPI::R_MAT(rand_graph, vertices_count, edges_count, 57, 19, 19, 5, direction_type);
            }
            else if(graph_type == "random_uniform" || graph_type == "ru" || graph_type == "random-uniform")
            {
                cout << "Generating random_uniform with: " << vertices_count << " vertices and " << edges_count << " edges" << endl;
                GraphGenerationAPI::random_uniform(rand_graph, vertices_count, edges_count, direction_type);
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
            rand_graph.remove_loops_and_multiple_arcs();

            VectCSRGraph vect_csr_graph;
            Timer tm;
            tm.start();
            vect_csr_graph.import(rand_graph);
            tm.end();
            tm.print_time_stats("Import");

            vect_csr_graph.print_stats();

            tm.start();
            add_extension(file_name, ".vgraph");
            vect_csr_graph.save_to_binary_file(file_name);
            cout << "VectCSR graph is generated and saved to file " << file_name << endl;
            tm.end();
            tm.print_time_stats("Save");

            //vect_csr_graph.print();
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
