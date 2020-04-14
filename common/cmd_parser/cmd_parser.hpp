#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

AlgorithmCommandOptionsParser::AlgorithmCommandOptionsParser()
{
    scale = 10;
    avg_degree = 5;
    compute_mode = GENERATE_NEW_GRAPH;
    algorithm_frontier_type = ALL_ACTIVE;
    traversal_direction = PUSH_TRAVERSAL;
    check_flag = false;
    graph_file_name = "test.gbin";
    number_of_rounds = 1;
    algorithm_bfs = DIRECTION_OPTIMISING_BFS_ALGORITHM;
    algorithm_cc = SHILOACH_VISHKIN_ALGORITHM;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void AlgorithmCommandOptionsParser::parse_args(int _argc, const char * _argv[])
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
            cout << "cmd parser selected all-active" << endl;
        }

        if (option.compare("-partial-active") == 0)
        {
            algorithm_frontier_type = PARTIAL_ACTIVE;
            cout << "cmd parser selected partial-active" << endl;
        }

        if (option.compare("-push") == 0)
        {
            traversal_direction = PUSH_TRAVERSAL;
            cout << "cmd parser selected push traversal" << endl;
        }

        if (option.compare("-pull") == 0)
        {
            traversal_direction = PULL_TRAVERSAL;
            cout << "cmd parser selected pull traversal" << endl;
        }

        if (option.compare("-top-down") == 0)
        {
            algorithm_bfs = TOP_DOWN_BFS_ALGORITHM;
        }
        else if (option.compare("-bottom-up") == 0)
        {
            algorithm_bfs = BOTTOM_UP_BFS_ALGORITHM;
        }
        else if (option.compare("-do") == 0 || option.compare("-direction-optimising") == 0)
        {
            algorithm_bfs = DIRECTION_OPTIMISING_BFS_ALGORITHM;
        }

        if ((option.compare("-shiloach_vishkin") == 0) || (option.compare("-sv") == 0))
        {
            algorithm_cc = SHILOACH_VISHKIN_ALGORITHM;
        }
        else if (option.compare("-bfs-based") == 0)
        {
            algorithm_cc = BFS_BASED_ALGORITHM;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
