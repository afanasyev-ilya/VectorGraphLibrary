#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::print()
{
    cout << endl;
    cout << "ExtendedCSRGraph format" << endl;

    cout << "|V|=" << this->vertices_count << endl;
    cout << "|E|=" << this->edges_count << endl;
    cout << "Vertices data: " << endl;
    for(int i = 0; i < this->vertices_count; i++)
        cout << this->vertex_values[i] << " ";
    cout << endl;

    cout << "Edges data: " << endl;
    for(int cur_vertex = 0; cur_vertex < this->vertices_count; cur_vertex++)
    {
        cout << "vertex " << cur_vertex << " connected to: ";
        for(long long edge_pos = vertex_pointers[cur_vertex]; edge_pos < vertex_pointers[cur_vertex + 1]; edge_pos++)
        {
            _TEdgeWeight weight = 0;

            cout << "(" << adjacent_ids[edge_pos] << "," << adjacent_weights[edge_pos] << ")" << " ";
        }
        cout << endl;
    }
    cout << endl;

    cout << "conversion array (original <--> current):" << endl;
    for(int i = 0; i < this->vertices_count; i++)
    {
        cout << i << " <--> " << forward_conversion[i] << endl;
    }
    cout << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
template <typename _TVertexValue, typename _TEdgeWeight>
void ExtendedCSRGraph<_TVertexValue, _TEdgeWeight>::print_with_weights(EdgesArrayNec<_TVertexValue, _TEdgeWeight, _TEdgeWeight> &_weights,
                                                                       TraversalDirection _direction)
{
    cout << endl;
    cout << "ExtendedCSRGraph format" << endl;

    cout << "|V|=" << this->vertices_count << endl;
    cout << "|E|=" << this->edges_count << endl;
    cout << "Vertices data: " << endl;
    for(int i = 0; i < this->vertices_count; i++)
        cout << this->vertex_values[i] << " ";
    cout << endl;

    cout << "Edges data: " << endl;
    for(int cur_vertex = 0; cur_vertex < this->vertices_count; cur_vertex++)
    {
        cout << "vertex " << cur_vertex << " connected to: ";
        for(long long edge_pos = vertex_pointers[cur_vertex]; edge_pos < vertex_pointers[cur_vertex + 1]; edge_pos++)
        {
            int dst_id = adjacent_ids[edge_pos];
            _TEdgeWeight weight = weights.get(cur_vertex, dst_id, _direction);

            cout << "(" << dst_id << "," << weight << ")" << " "; // TODO fix incoming case
        }
        cout << endl;
    }
    cout << endl;

    cout << "conversion array (original <--> current):" << endl;
    for(int i = 0; i < this->vertices_count; i++)
    {
        cout << i << " <--> " << forward_conversion[i] << endl;
    }
    cout << endl << endl;
}
*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
