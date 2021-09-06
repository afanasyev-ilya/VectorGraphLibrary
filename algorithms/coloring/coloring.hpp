#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void print(size_t _val)
{
    std::bitset<64> x(_val);
    std::cout << x << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline size_t set_bit(size_t _val, int _pos)
{
    _val |= 1ULL << _pos;
    return _val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline size_t clear_bit(size_t _val, int _pos)
{
    _val &= ~(1ULL << _pos);
    return _val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline int get_bit(size_t _val, int _pos)
{
    return (_val >> _pos) & 1ULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline int smallest_bit_pos(size_t _input)
{
    if(_input == 0)
        return -1;

    size_t x = _input & ~(_input-1);
    int ret=0, cmp = (x>(1LL<<31))<<5; //32 if true else 0
    ret += cmp;
    x  >>= cmp;
    cmp = (x>(1<<15))<<4; //16 if true else 0
    ret += cmp;
    x  >>= cmp;
    cmp = (x>(1<<7))<<3; //8
    ret += cmp;
    x  >>= cmp;
    cmp = (x>(1<<3))<<2; //4
    ret += cmp;
    x  >>= cmp;
    cmp = (x>(1<<1))<<1; //2
    ret += cmp;
    x  >>= cmp;
    cmp = (x>1);
    ret += cmp;
    x  >>= cmp;
    ret += x;
    return ret-1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
double Coloring::vgl_coloring(VGL_Graph &_graph, VerticesArray<_T> &_colors)
{
    VerticesArray<size_t> available_colors(_graph);
    VerticesArray<int> need_recolor(_graph);

    VGL_GRAPH_ABSTRACTIONS graph_API(_graph);
    VGL_FRONTIER frontier(_graph);

    graph_API.change_traversal_direction(SCATTER, frontier, _colors, available_colors, need_recolor);

    Timer tm;
    tm.start();
    frontier.set_all_active();
    auto init_op = [_colors, available_colors] __VGL_COMPUTE_ARGS__{
        _colors[src_id] = 0;
        available_colors[src_id] = -1;
    };
    graph_API.compute(_graph, frontier, init_op);

    int start_range = 0;
    int end_range = 64;

    int iterations = 0;
    while(frontier.size() > 0)
    {
        available_colors.set_all_constant(-1);
        auto mark_forbidden_op = [_colors, available_colors, start_range, end_range] __VGL_SCATTER_ARGS__ {
            int dst_color = _colors[dst_id];
            if((dst_color >= start_range) && (dst_color < end_range) && (src_id != dst_id))
            {
                size_t old_data = available_colors[src_id];
                available_colors[src_id] = clear_bit(old_data, dst_color - start_range);
            }
        };

        auto vertex_postprocess_op = [_colors, available_colors, start_range] __VGL_ADVANCE_POSTPROCESS_ARGS__ {
            int bit_pos = smallest_bit_pos(available_colors[src_id]);
            if(bit_pos >= 0)
                _colors[src_id] = bit_pos + start_range;
        };

        graph_API.enable_safe_stores();
        graph_API.scatter(_graph, frontier, mark_forbidden_op, EMPTY_VERTEX_OP, vertex_postprocess_op, mark_forbidden_op, EMPTY_VERTEX_OP, vertex_postprocess_op);
        graph_API.disable_safe_stores();

        need_recolor.set_all_constant(0);
        auto create_reordering_op = [_colors, need_recolor] __VGL_SCATTER_ARGS__ {
            if((_colors[dst_id] == _colors[src_id]) && (src_id != dst_id))
            {
                int max_id = vect_max(src_id, dst_id);
                need_recolor[max_id] = 1;
            }
        };
        graph_API.scatter(_graph, frontier, create_reordering_op);

        auto offset_change_required_op = [available_colors] __VGL_REDUCE_INT_ARGS__
        {
            int result = 0;
            if(available_colors[src_id] == 0)
                result = 1;
            return result;
        };

        int full_vertices = graph_API.reduce<int>(_graph, frontier, offset_change_required_op, REDUCE_SUM);
        if(full_vertices > 0)
        {
            start_range += 64;
            end_range += 64;
        }

        auto need_recolor_op = [need_recolor] __VGL_GNF_ARGS__ {
            int result = NOT_IN_FRONTIER_FLAG;
            if(need_recolor[src_id] == 1)
                result = IN_FRONTIER_FLAG;
            return result;
        };

        graph_API.generate_new_frontier(_graph, frontier, need_recolor_op);
        iterations++;
    }
    tm.end();

    #ifdef __PRINT_SAMPLES_PERFORMANCE_STATS__
    cout << "Iterations: " << iterations << endl;
    performance_stats.print_algorithm_performance_stats("Coloring (NEC/multicore)", tm.get_time(), _graph.get_edges_count());
    #endif

    return performance_stats.get_algorithm_performance(tm.get_time(), _graph.get_edges_count());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

