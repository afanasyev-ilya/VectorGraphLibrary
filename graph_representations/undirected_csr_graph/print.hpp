#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void UndirectedCSRGraph::print()
{
    cout << endl;
    cout << "UndirectedCSRGraph format" << endl;

    cout << "|V|=" << this->vertices_count << endl;
    cout << "|E|=" << this->edges_count << endl;
    for(int cur_vertex = 0; cur_vertex < this->vertices_count; cur_vertex++)
    {
        cout << "vertex " << cur_vertex << " connected to: ";
        for(long long edge_pos = vertex_pointers[cur_vertex]; edge_pos < vertex_pointers[cur_vertex + 1]; edge_pos++)
        {
            cout << "(" << adjacent_ids[edge_pos] << ")" << " ";
        }
        cout << endl;
    }
    cout << endl;

    cout << "conversion array (original --> sorted):" << endl;
    for(int i = 0; i < this->vertices_count; i++)
    {
        cout << i << " --> " << forward_conversion[i] << endl;
    }
    cout << endl << endl;

    cout << "conversion array (sorted --> original):" << endl;
    for(int i = 0; i < this->vertices_count; i++)
    {
        cout << i << " --> " << backward_conversion[i] << endl;
    }
    cout << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void UndirectedCSRGraph::print_with_weights(EdgesArray<_T> &_weights, TraversalDirection _direction)
{
    cout << endl;
    cout << "UndirectedCSRGraph format" << endl;

    cout << "|V|=" << this->vertices_count << endl;
    cout << "|E|=" << this->edges_count << endl;

    cout << "Edges data: " << endl;
    for(int cur_vertex = 0; cur_vertex < this->vertices_count; cur_vertex++)
    {
        cout << "vertex " << cur_vertex << " connected to: ";
        for(long long edge_pos = vertex_pointers[cur_vertex]; edge_pos < vertex_pointers[cur_vertex + 1]; edge_pos++)
        {
            int dst_id = adjacent_ids[edge_pos];
            _T weight = _weights.get(cur_vertex, dst_id, _direction);

            cout << "(" << dst_id << "," << weight << ")" << " "; // TODO fix incoming case
        }
        cout << endl;
    }
    cout << endl;

    cout << "conversion array (original --> sorted):" << endl;
    for(int i = 0; i < this->vertices_count; i++)
    {
        cout << i << " --> " << forward_conversion[i] << endl;
    }
    cout << endl << endl;

    cout << "conversion array (sorted --> original):" << endl;
    for(int i = 0; i < this->vertices_count; i++)
    {
        cout << i << " --> " << backward_conversion[i] << endl;
    }
    cout << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void UndirectedCSRGraph::print_size()
{
    cout << "Wall size (UndirectedCSRGraph): " << get_size()/1e9 << "GB" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t UndirectedCSRGraph::get_size()
{
    size_t size = 0;
    size += sizeof(vertex_pointers[0])*(this->vertices_count+1);
    size += sizeof(adjacent_ids[0])*(this->edges_count);
    size += 2*sizeof(forward_conversion[0])*this->vertices_count;
    size += sizeof(edges_reorder_indexes[0])*this->edges_count;
    size += last_vertices_ve.get_size();
    return size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void UndirectedCSRGraph::print_vertex_information(int _src_id, int _num_edges)
{
    cout << "vertex " << _src_id << " connected to: ";
    long long first = vertex_pointers[_src_id];
    long long last = vertex_pointers[_src_id + 1];
    for(long long edge_pos = 0; edge_pos < (last - first); edge_pos++)
    {
        int dst_id = adjacent_ids[first + edge_pos];
        if(edge_pos < _num_edges)
            cout << dst_id << " ";
    }
    cout << " (printed " << _num_edges << " edges, real count = " << last - first << ")" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
