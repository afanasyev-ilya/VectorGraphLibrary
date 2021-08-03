#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Parser::Parser()
{
    scale = 10;
    avg_degree = 5;
    graph_type = RMAT;
    compute_mode = GENERATE_NEW_GRAPH;
    algorithm_frontier_type = ALL_ACTIVE;
    traversal_direction = PUSH_TRAVERSAL;
    check_flag = false;
    graph_file_name = "test.graph";
    number_of_rounds = 1;
    algorithm_bfs = TOP_DOWN_BFS_ALGORITHM;
    algorithm_cc = SHILOACH_VISHKIN_ALGORITHM;
    graph_storage_format = VECTOR_CSR_GRAPH;

    device_num = 0;

    store_walk_paths = false;
    walk_vertices_percent = 1;
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
            graph_type = RANDOM_UNIFORM;
        }

        if ((option.compare("-rmat") == 0) || (option.compare("-RMAT") == 0))
        {
            graph_type = RMAT;
        }

        if ((option.compare("-type") == 0))
        {
            string tmp_type = _argv[++i];
            if((tmp_type == "rmat") || (tmp_type == "RMAT"))
                graph_type = RMAT;
            if((tmp_type == "random_uniform") || (tmp_type == "ru"))
                graph_type = RANDOM_UNIFORM;
        }

        if ((option.compare("-format") == 0))
        {
            string tmp_type = _argv[++i];
            if((tmp_type == "el") || (tmp_type == "edges_list"))
                graph_storage_format = EDGES_LIST_GRAPH;
            if((tmp_type == "vcsr") || (tmp_type == "vect_csr"))
                graph_storage_format = VECTOR_CSR_GRAPH;
            cout << "graph_storage_format: " << graph_storage_format << endl;
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
            algorithm_cc = SHILOACH_VISHKIN_ALGORITHM;
        }
        else if ((option.compare("-bfs-based") == 0) || (option.compare("-bfs_based") == 0))
        {
            algorithm_cc = BFS_BASED_ALGORITHM;
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
