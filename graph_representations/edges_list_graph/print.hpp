#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void EdgesListGraph<_TVertexValue, _TEdgeWeight>::print()
{
    cout << endl;
    cout << "Graph in edges list format" << endl;
    cout << "|V|=" << this->vertices_count << endl;
    cout << "|E|=" << this->edges_count << endl;
    cout << "Vertices data: " << endl;
    for(int i = 0; i < this->vertices_count; i++)
        cout << this->vertex_values[i] << " ";
    cout << endl;
    cout << "Edges data: " << endl;
    for(long long int i = 0; i < this->edges_count; i++)
        cout << src_ids[i] << " " << dst_ids[i] << " " << weights[i] << endl;
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void EdgesListGraph<_TVertexValue, _TEdgeWeight>::print_in_csr_format()
{
    this->preprocess_into_csr_based();

    cout << endl;
    cout << "Graph in edges list format" << endl;
    cout << "|V|=" << this->vertices_count << endl;
    cout << "|E|=" << this->edges_count << endl;
    cout << "Vertices data: " << endl;
    for(int i = 0; i < this->vertices_count; i++)
        cout << this->vertex_values[i] << " ";
    cout << endl;
    cout << "Edges data: " << endl;

    for(long long int i = 0; i < this->edges_count; i++)
    {
        cout << src_ids[i] << " -> " << dst_ids[i] << ", " << weights[i] << endl;
        if((i != (this->edges_count - 1)) && (src_ids[i] != src_ids[i + 1]))
            cout << endl;
    }

    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void EdgesListGraph<_TVertexValue, _TEdgeWeight>::print_stats()
{
    throw "print_stats not implemented yet";
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
