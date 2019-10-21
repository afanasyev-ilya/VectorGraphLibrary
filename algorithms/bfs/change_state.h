//
//  change_state.h
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 30/06/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef change_state_h
#define change_state_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum StateOfBFS
{
    TOP_DOWN,
    BOTTOM_UP
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

StateOfBFS change_state(int _current_queue_size, int _next_queue_size, int _vertices_count, long long _edges_count,
                        StateOfBFS _old_state, int _vis, int _in_lvl);
StateOfBFS gpu_change_state(int _current_queue_size, int _next_queue_size, int _vertices_count, long long _edges_count,
                            StateOfBFS _old_state, int _vis, int _in_lvl, int _current_level);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* change_state_h */
