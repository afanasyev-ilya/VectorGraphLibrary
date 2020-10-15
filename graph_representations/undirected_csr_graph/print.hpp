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
void UndirectedCSRGraph::print_with_weights(EdgesArrayNEC<_T> &_weights, TraversalDirection _direction)
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
    cout << "UndirectedCSRGraph size: " << get_size()/1e9 << "GB" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t UndirectedCSRGraph::get_size()
{
    size_t size = 0;
    size += sizeof(vertex_pointers[0])*(this->vertices_count+1);
    size += sizeof(adjacent_ids[0])*(this->edges_count);
    size += 2*sizeof(forward_conversion[0])*this->vertices_count;
    size += last_vertices_ve.get_size();
    return size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
