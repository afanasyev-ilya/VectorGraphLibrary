#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void EdgesListGraph::print()
{
    cout << endl;
    cout << "Graph in edges list format" << endl;
    cout << "|V|=" << this->vertices_count << endl;
    cout << "|E|=" << this->edges_count << endl;
    for(long long int i = 0; i < this->edges_count; i++)
        cout << src_ids[i] << " " << dst_ids[i] << endl;
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void EdgesListGraph::print_in_csr_format()
{
    this->preprocess_into_csr_based();

    cout << endl;
    cout << "Graph in edges list format" << endl;
    cout << "|V|=" << this->vertices_count << endl;
    cout << "|E|=" << this->edges_count << endl;
    for(long long int i = 0; i < this->edges_count; i++)
    {
        cout << src_ids[i] << " -> " << dst_ids[i] << endl;
        if((i != (this->edges_count - 1)) && (src_ids[i] != src_ids[i + 1]))
            cout << endl;
    }
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void EdgesListGraph::print_stats()
{
    throw "print_stats not implemented yet";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
