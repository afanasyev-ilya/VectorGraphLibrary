#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void EdgesListGraph::remove_loops_and_multiple_arcs()
{
    this->preprocess_into_csr_based();

    long long new_edges_count = 0;
    long long first = 0;
    for(long long edge_pos = 1; edge_pos < this->edges_count; edge_pos++)
    {
        if(src_ids[edge_pos] != src_ids[edge_pos - 1])
        {
            long long last = edge_pos;
            int connections_count = last - first;
            if(connections_count >= 1)
            {
                Sorter::sort(&(dst_ids[first]), NULL, connections_count, SORT_ASCENDING);

                int duplicates_count = 0;
                for(long long cur_edge = first + 1; cur_edge < last; cur_edge++)
                {
                    if((dst_ids[cur_edge] == dst_ids[cur_edge - 1]))
                        duplicates_count++;
                }

                new_edges_count += connections_count - duplicates_count;
            }

            first = last;
        }
    }

    cout << "remove loops and MA in edges list stats: " << new_edges_count << " / " << this->edges_count <<
        "(" << 100.0 *  new_edges_count / this->edges_count <<  ")" << endl;

    int *new_src_ids, *new_dst_ids;
    MemoryAPI::allocate_array(&new_src_ids, new_edges_count);
    MemoryAPI::allocate_array(&new_dst_ids, new_edges_count);

    long long copy_pos = 0;
    first = 0;
    for(long long edge_pos = 1; edge_pos < this->edges_count; edge_pos++)
    {
        if(src_ids[edge_pos] != src_ids[edge_pos - 1])
        {
            long long last = edge_pos;
            int connections_count = last - first;
            if(connections_count >= 1)
            {
                int duplicates_count = 0;
                for(long long cur_edge = first; cur_edge < last; cur_edge++)
                {
                    if(cur_edge == first)
                    {
                        new_src_ids[copy_pos] = src_ids[cur_edge];
                        new_dst_ids[copy_pos] = dst_ids[cur_edge];
                        copy_pos++;
                    }
                    else
                    {
                        if((dst_ids[cur_edge] == dst_ids[cur_edge - 1]))
                        {
                            continue;
                        }
                        else
                        {
                            new_src_ids[copy_pos] = src_ids[cur_edge];
                            new_dst_ids[copy_pos] = dst_ids[cur_edge];
                            copy_pos++;
                        }
                    }
                }
            }

            first = last;
        }
    }

    this->resize(this->vertices_count, new_edges_count);

    MemoryAPI::copy(this->src_ids, new_src_ids, new_edges_count);
    MemoryAPI::copy(this->dst_ids, new_dst_ids, new_edges_count);

    MemoryAPI::free_array(new_src_ids);
    MemoryAPI::free_array(new_dst_ids);

    /*vector<vector<int>> tmp_data(vertices_count);

    for(long long i = 0; i < this->edges_count; i++)
    {
        int src_id = src_ids[i];
        int dst_id = dst_ids[i];
        tmp_data[src_id].push_back(ds_id);
    }
    cout << " unroll into dict is done " << endl;

    for(long long cur_vertex = 0; cur_vertex < this->vertices_count; cur_vertex++)
    {
        int connections_count = tmp_data[src_id].size();
        if(connections_count >= 2)
        {
            Sorter::sort(&(tmp_data[src_id][0]), NULL, connections_count, SORT_ASCENDING);
        }
    }
    cout << " sort done " << endl;

    for(long long cur_vertex = 0; cur_vertex < this->vertices_count; cur_vertex++)
    {

    }*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void EdgesListGraph::import(EdgesContainer &_edges_container)
{
    this->resize(_edges_container.get_vertices_count(), _edges_container.get_edges_count());

    MemoryAPI::copy(this->src_ids, _edges_container.get_src_ids(), _edges_container.get_edges_count());
    MemoryAPI::copy(this->dst_ids, _edges_container.get_dst_ids(), _edges_container.get_edges_count());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
