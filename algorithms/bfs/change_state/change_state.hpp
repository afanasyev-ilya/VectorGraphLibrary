#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define ALPHA 15
#define BETA 18

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GraphStructure check_graph_structure(UndirectedCSRGraph &_graph)
{
    int vertices_count    = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count   ();
    long long    *vertex_pointers    = _graph.get_vertex_pointers   ();
    int          *adjacent_ids     = _graph.get_adjacent_ids    ();
    
    int portion_of_first_vertices = 0.01 * vertices_count + 1;
    long long number_of_edges_in_first_portion = vertex_pointers[portion_of_first_vertices];
    
    if((100.0 * number_of_edges_in_first_portion) / edges_count > POWER_LAW_EDGES_THRESHOLD)
        return POWER_LAW_GRAPH;
    else
        return UNIFORM_GRAPH;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool check_if_vector_extension_should_be_used(int _non_zero_vertices_count, int _not_visited_count, StateOfBFS _current_state)
{
    if(_current_state == BOTTOM_UP)
    {
        //cout << (100.0 * _not_visited_count) / _non_zero_vertices_count << " %" << endl;
        //if((100.0 * _not_visited_count) / _non_zero_vertices_count > )
        return true;
    }
    return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

StateOfBFS nec_change_state(int _current_queue_size, int _next_queue_size, int _vertices_count, long long _edges_count,
                            StateOfBFS _old_state, int _vis, int _in_lvl, bool &_use_vect_CSR_extension, int _cur_level,
                            GraphStructure _graph_structure, int *_levels, int _number_of_bu_steps)
{
    StateOfBFS new_state = _old_state;
    int factor = (_edges_count / _vertices_count) / 2;

    if(_current_queue_size < _next_queue_size) // growing phase
    {
        if(_old_state == TOP_DOWN)
        {
            if(((_levels[0] != UNVISITED_VERTEX) || (_levels[1] != UNVISITED_VERTEX)) && (_next_queue_size > 500))
            {
                new_state = BOTTOM_UP;
            }
            else
            {
                if(_in_lvl < ((_vertices_count - _vis) * factor + _vertices_count) / ALPHA)
                {
                    new_state = TOP_DOWN;
                }
                else
                {
                    new_state = BOTTOM_UP;
                }
            }
        }
    }
    else // shrinking phase
    {
        if(_old_state == BOTTOM_UP)
        {
            if((_next_queue_size < ((_vertices_count - _vis) * factor) / (factor * BETA)) && (_number_of_bu_steps > 1))
            {
                new_state = TOP_DOWN;
            }
            else
            {
                new_state = BOTTOM_UP;
            }
        }
    }

    _use_vect_CSR_extension = true;

    if(_number_of_bu_steps <= 1)
    {
        _use_vect_CSR_extension = true;
    }
    else
    {
        _use_vect_CSR_extension = false;
    }

    return new_state;
}

void parallel_chooseDirection(bool &currentDirection, long int sizeGraph, long int sizeFrontier, long int sizeNext, int alpha, int beta, int *_levels, int iter)
{
    int edgesToCheck;
    double branching_factor=(double) (sizeNext-sizeFrontier)/sizeFrontier;

    /*#pragma omp master
    {
        cout << "old dir: " << currentDirection << "\n";
    }*/

    //this is the case we the graph is growing
    if(currentDirection && branching_factor>0){
        edgesToCheck = sizeNext * branching_factor;
        currentDirection=(edgesToCheck<(sizeGraph*branching_factor/alpha));
        //Here the graph is shrinking
        //if(iter <= 2 && ((_levels[0] != UNVISITED_VERTEX)))
        //    currentDirection = false;
    }else if(!currentDirection && branching_factor<0){
        edgesToCheck=sizeFrontier;
        currentDirection=(sizeFrontier<(sizeGraph/beta));
    }

    /*#pragma omp master
    {
        cout << "new dir: " << currentDirection << "\n";
        cout << "Size of next frontier \t" << sizeNext << "\n";
        cout << "Branching Factor \t" << branching_factor << "\n";
        cout << "Graph Dimension \t" << sizeGraph << "\n";
        cout << "Number of edges to check \t" <<  edgesToCheck << "\n";
    }*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

StateOfBFS gpu_change_state(int _current_queue_size, int _next_queue_size, int _vertices_count, long long _edges_count,
                            StateOfBFS _old_state, int _vis, int _in_lvl, int _current_level, GraphStructure _graph_structure)
{
    StateOfBFS new_state = _old_state;
    int factor = (_edges_count / _vertices_count) / 2;
    
    if(_current_queue_size < _next_queue_size) // growing phase
    {
        if(_old_state == TOP_DOWN)
        {
            if(_in_lvl < ((_vertices_count - _vis) * factor + _vertices_count) / ALPHA)
            {
                new_state = TOP_DOWN;
            }
            else
            {
                new_state = BOTTOM_UP;
            }
        }
    }
    else // shrinking phase
    {
        if(_old_state == BOTTOM_UP)
        {
            if(_next_queue_size < ((_vertices_count - _vis) * factor + _vertices_count) / (factor * BETA))
            {
                new_state = TOP_DOWN;
            }
            else
            {
                new_state = BOTTOM_UP;
            }
        }
    }

    //if(_current_level == 1)
    //    new_state = BOTTOM_UP;
    
    //cout << "ch: " << _current_queue_size << " -> " << _next_queue_size << endl;
    
    return new_state;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
