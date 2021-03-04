#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierMulticore::print()
{
    #pragma omp master
    {
        throw "Error in FrontierMulticore::print : not implemented yet";
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FrontierMulticore::print_stats()
{
    #pragma omp master
    {
        UndirectedCSRGraph *current_direction_graph;
        if(graph_ptr->get_type() == VECT_CSR_GRAPH)
        {
            VectCSRGraph *vect_csr_ptr = (VectCSRGraph*)graph_ptr;
            current_direction_graph = vect_csr_ptr->get_direction_graph_ptr(direction);
        }
        else
        {
            throw "Error in FrontierMulticore::print_stats : unsupported graph type";
        }

        const int ve_threshold = current_direction_graph->get_vector_engine_threshold_vertex();
        const int vc_threshold = current_direction_graph->get_vector_core_threshold_vertex();
        const int vertices_count = current_direction_graph->get_vertices_count();

        if(type != ALL_ACTIVE_FRONTIER)
        {
            cout << "frontier status: " << get_frontier_status_string(type) << ": " << get_frontier_status_string(vector_engine_part_type) << "/" <<
                 get_frontier_status_string(vector_core_part_type) << "/" << get_frontier_status_string(collective_part_type) << endl;
            cout << "frontier size: " << current_size << " from " << max_size << ", " <<
                 (100.0 * current_size) / max_size << " %" << endl;
            if(ve_threshold > 0)
                cout << 100.0 * vector_engine_part_size / (ve_threshold) << " % active first part" << endl;
            cout << 100.0 * vector_core_part_size / (vc_threshold - ve_threshold) << " % active second part" << endl;
            cout << 100.0 * collective_part_size / (vertices_count - vc_threshold) << " % active third part" << endl;
        }
        else
        {
            cout << "frontier status: " << get_frontier_status_string(type) << endl;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
