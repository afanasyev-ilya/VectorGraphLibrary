//
//  cmd_parser.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 07/09/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef cmd_parser_hpp
#define cmd_parser_hpp

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
    steps_count = 1;
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

        if ((option.compare("-steps") == 0) || (option.compare("-steps-count") == 0) ||
            (option.compare("-source-num") == 0))
        {
            steps_count = atoi(_argv[++i]);
        }
        
        if ((option.compare("-check") == 0))
        {
            check_flag = true;
        }
        
        if ((option.compare("-nocheck") == 0))
        {
            check_flag = false;
        }
        
        if ((option.compare("-rounds") == 0))
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
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* cmd_parser_hpp */
