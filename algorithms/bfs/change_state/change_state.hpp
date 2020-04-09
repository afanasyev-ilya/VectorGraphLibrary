//
//  change_state.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 30/06/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define ALPHA 15
#define BETA 18

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
GraphStructure check_graph_structure(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph)
{
    int vertices_count    = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count   ();
    long long    *outgoing_ptrs    = _graph.get_outgoing_ptrs   ();
    int          *outgoing_ids     = _graph.get_outgoing_ids    ();
    
    int portion_of_first_vertices = 0.01 * vertices_count + 1;
    long long number_of_edges_in_first_portion = outgoing_ptrs[portion_of_first_vertices];
    
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
                            GraphStructure _graph_structure, int *_levels)
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
    
    if((_old_state == TOP_DOWN) && (_graph_structure == POWER_LAW_GRAPH) && (_cur_level <= 3))
    {
        int high_degree_was_visited = 0;
        for(int i = 0; i < 1; i++)
            if(_levels[i] != UNVISITED_VERTEX)
                high_degree_was_visited = 1;

        if(high_degree_was_visited)
            new_state = BOTTOM_UP;  // in the case of RMAT graph better switch to bottom up early
    }
    
    if((_graph_structure == POWER_LAW_GRAPH) && (_cur_level == 1)) // tofix
    {
        _use_vect_CSR_extension = true;
    }
    else
    {
        _use_vect_CSR_extension = false;
    }
    
    return new_state;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

StateOfBFS gpu_change_state(int _current_queue_size, int _next_queue_size, int _vertices_count, long long _edges_count,
                            StateOfBFS _old_state, int _vis, int _in_lvl, int _current_level, GraphStructure _graph_structure,
                            int _total_visited)
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
            
            if(((double)_next_queue_size) / _current_queue_size > BOTTOM_UP_FORCE_SWITCH_THRESHOLD_POWER_LOW_GRAPHS)
            {
                new_state = BOTTOM_UP;  // in the case of RMAT graph better switch to bottom up early
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
    
    //cout << "ch: " << _current_queue_size << " -> " << _next_queue_size << endl;
    
    return new_state;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
