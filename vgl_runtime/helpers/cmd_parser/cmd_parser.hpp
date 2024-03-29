#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Parser::Parser()
{
    scale = 10;
    avg_degree = 5;
    synthetic_graph_type = RMAT;
    compute_mode = GENERATE_NEW_GRAPH;
    algorithm_frontier_type = ALL_ACTIVE;
    traversal_direction = PUSH_TRAVERSAL;
    check_flag = false;
    graph_file_name = "test.graph";
    number_of_rounds = 1;
    algorithm_bfs = TOP_DOWN_BFS_ALGORITHM;
    algorithm_cc = CC_SHILOACH_VISHKIN_ALGORITHM;
    algorithm_tc = TC_PURDOMS_ALGORITHM;

    #ifdef __USE_GPU__
    graph_storage_format = CSR_VG_GRAPH;
    #else
    graph_storage_format = VECTOR_CSR_GRAPH;
    #endif

    graph_storage_optimizations = OPT_NONE;

    device_num = 0;

    store_walk_paths = false;
    walk_vertices_percent = 1;

    convert = false;
    convert_name = "";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool starts_with(string _full, string _pattern)
{
    size_t found = _full.find(_pattern);
    if(found == 0)
    {
        return true;
    }
    return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Parser::parse_args(int _argc, char **_argv)
{
    // get params from cmd line
    for (int i = 1; i < _argc; i++)
    {
        string option(_argv[i]);

        if ((option.compare("-load") == 0) || (option.compare("-file") == 0) || (option.compare("-f") == 0))
        {
            graph_file_name = _argv[++i];
            compute_mode = LOAD_GRAPH_FROM_FILE;
        }

        if ((option.compare("-import") == 0))
        {
            graph_file_name = _argv[++i];
            compute_mode = IMPORT_EDGES_CONTAINER;
        }

        if (option.compare("-convert") == 0)
        {
            convert_name = _argv[++i];
            convert = true;
        }
        
        if ((option.compare("-gen") == 0) || (option.compare("-generate") == 0))
        {
            compute_mode = GENERATE_NEW_GRAPH;
        }
        
        if ((option.compare("-scale") == 0) || (option.compare("-s") == 0))
        {
            scale = atoi(_argv[++i]);
        }

        if ((option.compare("-random_uniform") == 0) || (option.compare("-ru") == 0))
        {
            synthetic_graph_type = RANDOM_UNIFORM;
        }

        if ((option.compare("-rmat") == 0) || (option.compare("-RMAT") == 0))
        {
            synthetic_graph_type = RMAT;
        }

        if ((option.compare("-type") == 0))
        {
            string tmp_type = _argv[++i];
            if((tmp_type == "rmat") || (tmp_type == "RMAT"))
                synthetic_graph_type = RMAT;
            if((tmp_type == "random_uniform") || (tmp_type == "ru"))
                synthetic_graph_type = RANDOM_UNIFORM;
        }

        if ((option.compare("-format") == 0))
        {
            string tmp_type = _argv[++i];
            if(tmp_type == "el_container")
            {
                graph_storage_format = EDGES_CONTAINER;
            }
            else if(starts_with(tmp_type, "el"))
            {
                graph_storage_format = EDGES_LIST_GRAPH;
                if(tmp_type == "el")
                {
                    graph_storage_optimizations = OPT_NONE;
                }
                else if(tmp_type == "el_2D_seg")
                {
                    graph_storage_optimizations = EL_2D_SEGMENTED;
                }
                else if(tmp_type == "el_csr_based")
                {
                    graph_storage_optimizations = EL_CSR_BASED;
                }
                else
                {
                    throw "Error in Parser::parse_args : unknown edges list graph storage format";
                }
            }
            else if((tmp_type == "vcsr") || (tmp_type == "vect_csr"))
            {
                graph_storage_format = VECTOR_CSR_GRAPH;
            }
            else if(tmp_type == "csr")
            {
                graph_storage_format = CSR_GRAPH;
            }
            else if(tmp_type == "csr_vg" || tmp_type == "vg_csr")
            {
                graph_storage_format = CSR_VG_GRAPH;
            }
            else
            {
                throw "Error in Parser::parse_args : unknown graph_storage_format";
            }
        }

        if ((option.compare("-edges") == 0) || (option.compare("-e") == 0))
        {
            avg_degree = atoi(_argv[++i]);
        }
        
        if ((option.compare("-check") == 0))
        {
            check_flag = true;
        }
        
        if ((option.compare("-nocheck") == 0))
        {
            check_flag = false;
        }
        
        if ((option.compare("-rounds") == 0) || (option.compare("-iterations") == 0) || (option.compare("-it") == 0))
        {
            number_of_rounds = atoi(_argv[++i]);
        }

        if (option.compare("-all-active") == 0)
        {
            algorithm_frontier_type = ALL_ACTIVE;
        }

        if (option.compare("-partial-active") == 0)
        {
            algorithm_frontier_type = PARTIAL_ACTIVE;
        }

        if (option.compare("-push") == 0)
        {
            traversal_direction = PUSH_TRAVERSAL;
        }

        if (option.compare("-pull") == 0)
        {
            traversal_direction = PULL_TRAVERSAL;
        }

        if ((option.compare("-top-down") == 0) || (option.compare("-td") == 0))
        {
            algorithm_bfs = TOP_DOWN_BFS_ALGORITHM;
        }
        else if ((option.compare("-bottom-up") == 0) || (option.compare("-bu") == 0))
        {
            algorithm_bfs = BOTTOM_UP_BFS_ALGORITHM;
        }
        else if ((option.compare("-direction-optimizing") == 0) || (option.compare("-do") == 0))
        {
            algorithm_bfs = DIRECTION_OPTIMIZING_BFS_ALGORITHM;
        }

        if ((option.compare("-shiloach-vishkin") == 0) || (option.compare("-sv") == 0))
        {
            algorithm_cc = CC_SHILOACH_VISHKIN_ALGORITHM;
        }
        else if ((option.compare("-bfs-based") == 0) || (option.compare("-bfs_based") == 0))
        {
            algorithm_cc = CC_BFS_BASED_ALGORITHM;
            algorithm_tc = TC_BFS_BASED_ALGORITHM;
        }
        else if((option.compare("-purdoms") == 0) || (option.compare("-purdom") == 0))
        {
            algorithm_tc = TC_PURDOMS_ALGORITHM;
        }

        if ((option.compare("-store-walk-paths") == 0))
        {
            store_walk_paths = true;
            cout << "store_walk_paths set to TRUE" << endl;
        }

        if ((option.compare("-walk-vertices") == 0) || (option.compare("-wv") == 0))
        {
            walk_vertices_percent = atoi(_argv[++i]);
        }

        if ((option.compare("-dev") == 0) || (option.compare("-device") == 0))
        {
            device_num = atoi(_argv[++i]);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TraversalDirection Parser::convert_traversal_type(AlgorithmTraversalType _algo_type)
{
    if(_algo_type == PUSH_TRAVERSAL)
        return SCATTER;
    if(_algo_type == PULL_TRAVERSAL)
        return GATHER;

    throw "Error in Parser::convert_traversal_type : incorrect _algo_type value";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

AlgorithmTraversalType Parser::convert_traversal_type(TraversalDirection _direction_type)
{
    if(_direction_type == SCATTER)
        return PUSH_TRAVERSAL;
    if(_direction_type == GATHER)
        return PULL_TRAVERSAL;

    throw "Error in Parser::convert_traversal_type : incorrect _direction_type value";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
