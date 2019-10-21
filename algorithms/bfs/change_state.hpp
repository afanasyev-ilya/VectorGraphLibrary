//
//  change_state.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 30/06/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef change_state_hpp
#define change_state_hpp

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define ALPHA 15
#define BETA 18

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

StateOfBFS change_state(int _current_queue_size, int _next_queue_size, int _vertices_count, long long _edges_count,
                        StateOfBFS _old_state, int _vis, int _in_lvl)
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
    
    return new_state;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

StateOfBFS gpu_change_state(int _current_queue_size, int _next_queue_size, int _vertices_count, long long _edges_count,
                            StateOfBFS _old_state, int _vis, int _in_lvl, int _current_level)
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
    
    //if(_current_level == 2)
    //    new_state = BOTTOM_UP;
    
    return new_state;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* change_state_h */
