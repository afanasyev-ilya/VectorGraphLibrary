#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void save_flows_to_graphviz_file(VGL_Graph &_graph,
                                 EdgesArray<_T> &_weights,
                                 EdgesArray<_T> &_flows,
                                 int _source,
                                 int _sink,
                                 string _file_name)
{
    ofstream dot_output(_file_name.c_str());

    string connection;
    dot_output << "digraph G {" << endl;
    connection = " -> ";

    int vertices_count = _graph.get_vertices_count();
    for(int src_id = 0; src_id < vertices_count; src_id++)
    {
        if(src_id == _source)
            dot_output << src_id << "[label=\"" << "source" << "\"];" << endl;
        if(src_id == _sink)
            dot_output << src_id << "[label=\"" << "sink" << "\"];" << endl;
        int connections_count = _graph.get_outgoing_connections_count(src_id);
        for(int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            int dst_id = _graph.get_outgoing_edge_dst(src_id, edge_pos);
            _T orig_weight = _weights[_graph.get_outgoing_edges_array_index(src_id, edge_pos)];
            _T flow = _flows[_graph.get_outgoing_edges_array_index(src_id, edge_pos)];

            if(flow > orig_weight) // prevent printing increased flows (e.g. 200/100)
                dot_output << src_id << connection << dst_id << " [label = \" " << orig_weight << "/" << orig_weight << " \"];" << endl;
            else
                dot_output << src_id << connection << dst_id << " [label = \" " << flow << "/" << orig_weight << " \"];" << endl;
        }
    }

    dot_output << "}";
    dot_output.close();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

