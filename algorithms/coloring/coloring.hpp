#pragma once

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

void print(size_t _val)
{
    std::bitset<64> x(_val);
    std::cout << x << '\n';
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Coloring::vgl_coloring(VectCSRGraph &_graph)
{
    cout << "here" << endl;

    VerticesArray<size_t> available_colors(_graph);
    VerticesArray<int> colors(_graph);
    VerticesArray<int> need_recolor(_graph);

    VGL_GRAPH_ABSTRACTIONS graph_API(_graph);
    VGL_FRONTIER frontier(_graph);

    frontier.set_all_active();

    auto init_op = [colors, available_colors] __VGL_COMPUTE_ARGS__{
        colors[src_id] = 0;
        available_colors[src_id] = -1;
    };
    graph_API.compute(_graph, frontier, init_op);

    int it = 0;
    while(frontier.size() > 0)
    {
        auto mark_forbidden_op = [colors, available_colors] __VGL_SCATTER_ARGS__ {
            int dst_color = colors[dst_id];
            size_t cur_data = available_colors[src_id];
            available_colors[src_id] = clear_bit(cur_data, dst_color);
        };

        graph_API.scatter(_graph, frontier, mark_forbidden_op);

        print(available_colors[0]);
        //print(available_colors[1]);
        cout << "prev: " << colors[0] << endl;

        auto recolor_op = [colors, available_colors] __VGL_COMPUTE_ARGS__{
            size_t cur_data = available_colors[src_id];
            #pragma _NEC unroll(64)
            for(int i = 0; i < 64; i++)
            {
                int bit = get_bit(cur_data, i);
                if(bit == 1)
                {
                    colors[src_id] = i;
                    break;
                }
            }
        };
        graph_API.compute(_graph, frontier, recolor_op);

        cout << "new: " << colors[0] << endl;

        need_recolor.set_all_constant(0);
        auto create_reordering_op = [colors, need_recolor] __VGL_SCATTER_ARGS__ {
            if(colors[dst_id] == colors[src_id])
            {
                int min_id = vect_min(src_id, dst_id);
                need_recolor[min_id] = 1;
                //int max_id = vect_max(src_id, dst_id); // TODO this is correct
                //need_recolor[max_id] = 1;
            }
        };
        graph_API.scatter(_graph, frontier, create_reordering_op);

        auto need_recolor_op = [need_recolor] __VGL_GNF_ARGS__ {
            int result = NOT_IN_FRONTIER_FLAG;
            if(need_recolor[src_id] == 1)
                result = IN_FRONTIER_FLAG;
            return result;
        };

        graph_API.generate_new_frontier(_graph, frontier, need_recolor_op);

        cout << frontier.size() << " / " << _graph.get_vertices_count() << endl;
        it++;
        if(it > 50)
            break;
    }

    int connections_count = _graph.get_outgoing_connections_count(0);
    for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
    {
        int v = _graph.get_outgoing_edge_dst(0, edge_pos);
        cout << colors[v] << " ";
    }
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

