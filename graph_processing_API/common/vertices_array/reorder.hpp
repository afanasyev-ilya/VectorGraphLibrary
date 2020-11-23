#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VerticesArray<_T>::reorder(TraversalDirection _output_dir)
{
    if(graph_ptr->get_type() == VECT_CSR_GRAPH)
    {
        VectCSRGraph *vect_ptr = (VectCSRGraph *)graph_ptr;
        vect_ptr->reorder(*this, _output_dir);
    }
    if(graph_ptr->get_type() == SHARDED_CSR_GRAPH)
    {
        if((_output_dir != ORIGINAL) && (this->direction != ORIGINAL))
        {
            throw "Error in VerticesArray<_T>::reorder : SHARDED_CSR_GRAPH reorder wrong directions, can be ORIGINAL only";
        }
    }
    if(graph_ptr->get_type() == EDGES_LIST_GRAPH)
    {
        if((_output_dir != ORIGINAL) && (this->direction != ORIGINAL))
        {
            throw "Error in VerticesArray<_T>::reorder : EDGES_LIST_GRAPH reorder wrong directions, can be ORIGINAL only";
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VerticesArray<_T>::reorder_from_original_to_shard(TraversalDirection _direction, int _shard_id)
{
    if(graph_ptr->get_type() != SHARDED_CSR_GRAPH)
    {
        throw "Error in VerticesArray<_T>::reorder_to_shard : incorrect graph type for vertex array";
    }

    ShardedCSRGraph *sharded_graph_ptr = (ShardedCSRGraph *)graph_ptr;
    sharded_graph_ptr->reorder_from_original_to_shard(*this, _direction, _shard_id);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void VerticesArray<_T>::reorder_from_shard_to_original(TraversalDirection _direction, int _shard_id)
{
    if(graph_ptr->get_type() != SHARDED_CSR_GRAPH)
    {
        throw "Error in VerticesArray<_T>::reorder_to_original : incorrect graph type for vertex array";
    }

    ShardedCSRGraph *sharded_graph_ptr = (ShardedCSRGraph *)graph_ptr;
    sharded_graph_ptr->reorder_from_shard_to_original(*this, _direction, _shard_id);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
