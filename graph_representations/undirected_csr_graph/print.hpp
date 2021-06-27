#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void UndirectedVectCSRGraph::print()
{
    cout << endl;
    cout << "UndirectedVectCSRGraph format" << endl;

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
void UndirectedVectCSRGraph::print_with_weights(EdgesArray<_T> &_weights, TraversalDirection _direction)
{
    cout << endl;
    cout << "UndirectedVectCSRGraph format" << endl;

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

void UndirectedVectCSRGraph::print_size()
{
    cout << "Wall size (UndirectedVectCSRGraph): " << get_size()/1e9 << " GB" << endl;
    cout << "     CSR size: " << get_csr_size()/1e9 << " GB" << endl;
    cout << "     VE  size: " << get_ve_size()/1e9 << " GB" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t UndirectedVectCSRGraph::get_size()
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

size_t UndirectedVectCSRGraph::get_csr_size()
{
    size_t size = 0;
    size += sizeof(vertex_pointers[0])*(this->vertices_count+1);
    size += sizeof(adjacent_ids[0])*(this->edges_count);
    return size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t UndirectedVectCSRGraph::get_ve_size()
{
    return last_vertices_ve.get_size() + sizeof(edges_reorder_indexes[0])*this->edges_count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void UndirectedVectCSRGraph::print_vertex_information(int _src_id, int _num_edges)
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
    cout << " (printed " << min((int)_num_edges, (int)(last - first)) << " edges, real count = " << last - first << ")" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void UndirectedVectCSRGraph::print_stats()
{
    #ifdef __USE_NEC_SX_AURORA__
    cout << "threshold vertices: " << vector_engine_threshold_vertex << " " << vector_core_threshold_vertex << " " << vertices_count << endl;

    long long edges_in_ve_group = vertex_pointers[vector_engine_threshold_vertex] - vertex_pointers[0];
    long long edges_in_vc_group = vertex_pointers[vector_core_threshold_vertex] - vertex_pointers[vector_engine_threshold_vertex];
    long long edges_in_collective_group = vertex_pointers[this->vertices_count] - vertex_pointers[vector_core_threshold_vertex];

    cout << "ve group size: " << edges_in_ve_group << ", " << 100.0*((float)edges_in_ve_group)/this->edges_count << " %" << endl;
    cout << "vc group size: " << edges_in_vc_group << ", " << 100.0*((float)edges_in_vc_group)/this->edges_count << " %" << endl;
    cout << "collective group size: " << edges_in_collective_group << ", " << 100.0*((float)edges_in_collective_group)/this->edges_count << " %" << endl;
    #endif

    GraphAnalytics::analyse_degrees(*this);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

