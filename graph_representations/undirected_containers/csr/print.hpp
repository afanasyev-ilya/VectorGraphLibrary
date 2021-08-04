#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::print()
{
    cout << endl;
    cout << "CSRGraph format" << endl;

    cout << "|V|=" << this->vertices_count << endl;
    cout << "|E|=" << this->edges_count << endl;
    for(int cur_vertex = 0; cur_vertex < this->vertices_count; cur_vertex++)
    {
        cout << "vertex " << cur_vertex << " connected to: " << vertex_pointers[cur_vertex + 1] - vertex_pointers[cur_vertex] << " vertices";
        /*cout << "vertex " << cur_vertex << " connected to: ";
        for(long long edge_pos = vertex_pointers[cur_vertex]; edge_pos < vertex_pointers[cur_vertex + 1]; edge_pos++)
        {
            cout << "(" << adjacent_ids[edge_pos] << ")" << " ";
        }*/
        cout << endl;
    }
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CSRGraph::print_size()
{
    cout << "Wall size (CSRGraph): " << get_size()/1e9 << " GB" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t CSRGraph::get_size()
{
    size_t size = 0;
    size += sizeof(vertex_pointers[0])*(this->vertices_count+1);
    size += sizeof(adjacent_ids[0])*(this->edges_count);
    return size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
