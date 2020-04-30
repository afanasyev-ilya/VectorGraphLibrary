#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename _TVertexValue, typename _TEdgeWeight>
void always_active(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                   GraphPrimitivesGPU &graph_API,
                   FrontierGPU &frontier,
                   int *new_ptr,
                   int *gathered_labels,
                   int *_labels,
                   int *updated,
                   int *node_states,
                   int *seg_reduce_result,
                   int *reduced_scan,
                   int *seg_reduce_indices,
                   int _iterations_count)
{
    auto get_labels_op = [seg_reduce_result, reduced_scan, gathered_labels, _labels, updated] __device__(int src_id, int position_in_frontier, int connections_count)
    {
        if (seg_reduce_result[position_in_frontier] != -1)
        {
            int new_label = gathered_labels[reduced_scan[seg_reduce_result[position_in_frontier]]];

            if (new_label != _labels[src_id])
            {
                _labels[src_id] = new_label;
                updated[0] = 1;
            }
        }
    };

    graph_API.compute(_graph, frontier, get_labels_op);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename _TVertexValue, typename _TEdgeWeight>
void active_passive_inner(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                          GraphPrimitivesGPU &graph_API,
                          FrontierGPU &frontier,
                          int *new_ptr,
                          int *gathered_labels,
                          int *_labels,
                          int *updated,
                          int *node_states,
                          int *seg_reduce_result,
                          int *reduced_scan,
                          int *seg_reduce_indices,
                          int _iterations_count)
{
    int *changes_recently_occurred = new_ptr;

    auto get_labels_op = [seg_reduce_result, reduced_scan, gathered_labels, _labels, updated, node_states, changes_recently_occurred] __device__(int src_id, int position_in_frontier, int connections_count)
    {
        changes_recently_occurred[src_id] = 0;

        if(seg_reduce_result[position_in_frontier] != -1)
        {
            int new_label = gathered_labels[reduced_scan[seg_reduce_result[position_in_frontier]]];

            if(node_states[src_id] == LP_BOUNDARY_ACTIVE)
            {
                if(new_label != _labels[src_id])
                {
                    _labels[src_id] = new_label;
                    updated[0] = 1;
                    node_states[src_id] = LP_BOUNDARY_ACTIVE;
                    changes_recently_occurred[src_id] = 1;
                }

                if (new_label == _labels[src_id])
                {
                    node_states[src_id] = LP_BOUNDARY_PASSIVE;
                }
            }
        }
    };

    graph_API.compute(_graph, frontier, get_labels_op);

    auto label_recently_changed = [changes_recently_occurred] __device__ (int src_id)->int
    {
        if(changes_recently_occurred[src_id] > 0)
            return IN_FRONTIER_FLAG;
        else
            return NOT_IN_FRONTIER_FLAG;
    };

    graph_API.generate_new_frontier(_graph, frontier, label_recently_changed);

    int *different_presence = seg_reduce_indices;
    auto preprocess_op = [different_presence] __device__(int src_id, int position_in_frontier, int connections_count)
    {
        different_presence[src_id] = 0;
    };

    auto postprocess_op = [different_presence, node_states] __device__(int src_id, int position_in_frontier, int connections_count)
    {
        if(different_presence[src_id] == 0)
            node_states[src_id] = LP_INNER;
    };

    auto set_all_neighbours_active = [_labels, node_states, different_presence] __device__(int src_id, int dst_id, int local_edge_pos, long long int global_edge_pos, int position_in_frontier)
    {
        int dst_label = __ldg(&_labels[dst_id]);
        int src_label = _labels[src_id];

        if(src_label != dst_label)
            different_presence[src_id] = 1;

        if(node_states[dst_id] == LP_BOUNDARY_PASSIVE)
            node_states[dst_id] = LP_BOUNDARY_ACTIVE;
    };

    graph_API.advance(_graph, frontier, set_all_neighbours_active, preprocess_op, postprocess_op);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename _TVertexValue, typename _TEdgeWeight>
void label_changed_on_previous_iteration(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                         GraphPrimitivesGPU &graph_API,
                                         FrontierGPU &frontier,
                                         int *new_ptr,
                                         int *gathered_labels,
                                         int *_labels,
                                         int *updated,
                                         int *node_states,
                                         int *seg_reduce_result,
                                         int *reduced_scan,
                                         int *seg_reduce_indices,
                                         int _iterations_count)
{
    auto get_labels_op = [seg_reduce_result, reduced_scan, gathered_labels, _labels, updated, node_states] __device__(int src_id, int position_in_frontier, int connections_count)
    {
        if(seg_reduce_result[position_in_frontier] != -1)
        {
            int new_label = gathered_labels[reduced_scan[seg_reduce_result[position_in_frontier]]];

            if(new_label != _labels[src_id])
            {
                _labels[src_id] = new_label;
                updated[0] = 1;
                node_states[src_id] = LP_BOUNDARY_ACTIVE;
            }

            if (new_label == _labels[src_id])
            {
                node_states[src_id] = LP_BOUNDARY_PASSIVE;
            }
        }
    };

    graph_API.compute(_graph, frontier, get_labels_op);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename _TVertexValue, typename _TEdgeWeight>
void label_changed_recently(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                            GraphPrimitivesGPU &graph_API,
                            FrontierGPU &frontier,
                            int *new_ptr,
                            int *gathered_labels,
                            int *_labels,
                            int *updated,
                            int *node_states,
                            int *seg_reduce_result,
                            int *reduced_scan,
                            int *seg_reduce_indices,
                            int _iterations_count)
{
    auto get_labels_op = [seg_reduce_result, reduced_scan, gathered_labels, _labels, updated, node_states] __device__(int src_id, int position_in_frontier, int connections_count)
    {
        if(seg_reduce_result[position_in_frontier] != -1)
        {
            int new_label = gathered_labels[reduced_scan[seg_reduce_result[position_in_frontier]]];

            if(new_label != _labels[src_id])
            {
                _labels[src_id] = new_label;
                updated[0] = 1;
                node_states[src_id]++;
                if(node_states[src_id] > 4)
                    node_states[src_id] = 4;
            }

            if (new_label == _labels[src_id])
            {
                node_states[src_id]--;
                if(node_states[src_id] < 0)
                    node_states[src_id] = 0;
            }
        }
    };

    graph_API.compute(_graph, frontier, get_labels_op);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

