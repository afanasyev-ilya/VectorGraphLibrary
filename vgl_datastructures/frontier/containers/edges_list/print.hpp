#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierEdgesList::print()
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

void FrontierEdgesList::print_stats()
{

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
