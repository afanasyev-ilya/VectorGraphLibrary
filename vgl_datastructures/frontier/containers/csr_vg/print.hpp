#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierCSR_VG::print()
{
    cout << "Frontier: ";
    if(this->sparsity_type == ALL_ACTIVE_FRONTIER)
    {
        for(int src_id = 0; src_id < this->size; src_id++)
        {
            cout << src_id << " ";
        }
        cout << endl;
    }
    else
    {
        for(int src_id = 0; src_id < this->size; src_id++)
        {
            if(this->flags[src_id] > 0)
            {
                cout << src_id << " ";
            }
        }
        cout << endl;
    }
    cout << "(" << this->size << " - frontier size)" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierCSR_VG::print_stats()
{

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierCSR_VG::print_vertex_group_sizes()
{
    for(int i = 0; i < CSR_VERTEX_GROUPS_NUM; i++)
        cout << "group " << i << " has size : " << vertex_groups[i].get_size() << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
