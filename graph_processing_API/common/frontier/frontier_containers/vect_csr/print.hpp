#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierVectorCSR::print()
{
    cout << "Frontier: ";
    if(this->type == ALL_ACTIVE_FRONTIER)
    {
        int frontier_size = this->max_size;
        for(int src_id = 0; src_id < frontier_size; src_id++)
        {
            cout << src_id << " ";
        }
        cout << endl;
    }
    else
    {
        int frontier_size = this->max_size;
        for(int src_id = 0; src_id < frontier_size; src_id++)
        {
            if(this->flags[src_id] > 0)
            {
                cout << src_id << " ";
            }
        }
        cout << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierVectorCSR::print_stats()
{

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
