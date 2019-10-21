//
//  verifier.hpp
//  ParallelGraphLibrary
//
//  Created by Elijah Afanasiev on 17/06/2019.
//  Copyright © 2019 MSU. All rights reserved.
//

#ifndef verifier_h
#define verifier_h

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::verifier(VectorisedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                int _source_vertex,
                                                int *_parallel_levels)
{
    LOAD_VECTORISED_CSR_GRAPH_DATA(_graph)
    
    vector<int> seq_levels(vertices_count);
    
    vector<int> to_visit;
    to_visit.reserve(vertices_count);
    to_visit.push_back(_source_vertex);
    
    for(int i = 0; i < vertices_count; i++)
        seq_levels[i] = -1;
    seq_levels[_source_vertex] = 1;
    
    for (int it = 0; it < to_visit.size(); it++)
    {
        int u = to_visit[it];
            
        long long edge_pos = _graph.get_vertex_pointer(u);
        int connections_count = _graph.get_vector_connections_count(u);
            
        int multiplier = 1;
        if(u >= number_of_vertices_in_first_part)
            multiplier = VECTOR_LENGTH;
            
        for(int i = 0; i < connections_count; i++)
        {
            int v = outgoing_ids[edge_pos + i * multiplier];
            if (seq_levels[v] == -1)
            {
                seq_levels[v] = seq_levels[u] + 1;
                to_visit.push_back(v);
            }
        }
    }
    
    int error_count = 0;
    for(int i = 0; i < vertices_count; i++)
    {
        if(_parallel_levels[i] != seq_levels[i])
        {
            if(error_count < 10)
            {
                cout << "error in " << i << ": " << _parallel_levels[i] << " " << seq_levels[i] << endl;
            }
            error_count++;
        }
    }
    
    cout << "error count in verifier: " << error_count << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::verifier(ExtendedCSRGraph<_TVertexValue, _TEdgeWeight> &_graph,
                                                int _source_vertex,
                                                int *_parallel_levels)
{
    int vertices_count    = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count   ();
    long long    *outgoing_ptrs    = _graph.get_outgoing_ptrs   ();
    int          *outgoing_ids     = _graph.get_outgoing_ids    ();
    _TEdgeWeight *outgoing_weights = _graph.get_outgoing_weights();
    
    vector<int> seq_levels(vertices_count);
    
    vector<int> to_visit;
    to_visit.reserve(vertices_count);
    to_visit.push_back(_source_vertex);
    
    for(int i = 0; i < vertices_count; i++)
        seq_levels[i] = -1;
    seq_levels[_source_vertex] = 1;
    
    for (int it = 0; it < to_visit.size(); it++)
    {
        int u = to_visit[it];
            
        long long edge_pos = outgoing_ptrs[u];
        int connections_count = outgoing_ptrs[u + 1] - outgoing_ptrs[u];
            
        for(int i = 0; i < connections_count; i++)
        {
            int v = outgoing_ids[edge_pos + i];
            if (seq_levels[v] == -1)
            {
                seq_levels[v] = seq_levels[u] + 1;
                to_visit.push_back(v);
            }
        }
    }
    
    int error_count = 0, correct_count = 0;
    for(int i = 0; i < vertices_count; i++)
    {
        if(_parallel_levels[i] != seq_levels[i])
        {
            error_count++;
        }
        else
        {
            correct_count++;
        }
    }
    
    cout << "correct count in verifier: " << correct_count << "/" << vertices_count << endl;
    cout << "error count in verifier: " << error_count << "/" << vertices_count << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* verifier_h */
