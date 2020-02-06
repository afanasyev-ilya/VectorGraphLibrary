#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class VectorExtension
{
private:
    int vertices_count;
    int starting_vertex;
    int vector_segments_count;

    // vector segments data
    long long *vector_group_ptrs;
    int *vector_group_sizes;

    // outgoing edges data
    int *adjacent_ids;
    _TEdgeWeight *adjacent_weights;
public:
    VectorExtension();
    ~VectorExtension();

    int get_vertices_count() {return vertices_count;};
    int get_starting_vertex() {return starting_vertex;};
    int get_vector_segments_count() {return vector_segments_count;};

    long long *get_vector_group_ptrs() {return vector_group_ptrs;};
    int *get_vector_group_sizes() {return vector_group_sizes;};
    int *get_adjacent_ids() {return adjacent_ids;};
    _TEdgeWeight *get_adjacent_weights() {return adjacent_weights;};

    void init_from_graph(vector<vector<TempEdgeData<_TEdgeWeight> > >&_tmp_graph, int _first_vertex, int _last_vertex);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
VectorExtension<_TVertexValue, _TEdgeWeight>::VectorExtension()
{
    vertices_count = VECTOR_LENGTH;
    starting_vertex = 0;
    vector_segments_count = 1;

    vector_group_ptrs = new long long[vector_segments_count];
    vector_group_sizes = new int[vector_segments_count];

    adjacent_ids = new int[VECTOR_LENGTH];
    adjacent_weights = new _TEdgeWeight[VECTOR_LENGTH];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
VectorExtension<_TVertexValue, _TEdgeWeight>::~VectorExtension()
{
    delete []vector_group_ptrs;
    delete []vector_group_sizes;

    delete []adjacent_ids;
    delete []adjacent_weights;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void VectorExtension<_TVertexValue, _TEdgeWeight>::init_from_graph(vector<vector<TempEdgeData<_TEdgeWeight> > >&_tmp_graph,
                                                                   int _first_vertex,
                                                                   int _last_vertex)
{
    vertices_count = _last_vertex - _first_vertex;
    starting_vertex = _first_vertex;
    vector_segments_count = (vertices_count - 1) / VECTOR_LENGTH + 1;

    long long edges_count = 0;
    for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
    {
        int vec_start = cur_vector_segment * VECTOR_LENGTH + starting_vertex;
        int cur_max_connections_count = _tmp_graph[vec_start].size();
        edges_count += cur_max_connections_count *VECTOR_LENGTH;
    }

    delete[] vector_group_ptrs;
    vector_group_ptrs  = new long long[vector_segments_count];
    delete[] vector_group_sizes;
    vector_group_sizes = new int[vector_segments_count];
    delete[] adjacent_ids;
    adjacent_ids = new int[edges_count + VECTOR_LENGTH];
    delete[] adjacent_weights;
    adjacent_weights = new _TEdgeWeight[edges_count + VECTOR_LENGTH];

    long long current_edge = 0;
    for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
    {
        int vec_start = cur_vector_segment * VECTOR_LENGTH + starting_vertex;
        int cur_max_connections_count = _tmp_graph[vec_start].size();

        vector_group_ptrs[cur_vector_segment] = current_edge;
        vector_group_sizes[cur_vector_segment] = cur_max_connections_count;

        for(int edge_pos = 0; edge_pos < cur_max_connections_count; edge_pos++)
        {
            #pragma unroll
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = vec_start + i;
                if((src_id < _last_vertex) && (edge_pos < _tmp_graph[src_id].size()))
                {
                    adjacent_ids[current_edge + i] = _tmp_graph[src_id][edge_pos].dst_id;
                    adjacent_weights[current_edge + i] = _tmp_graph[src_id][edge_pos].weight;
                }
                else
                {
                    adjacent_ids[current_edge + i] = src_id;
                    adjacent_weights[current_edge + i] = 0.0;
                }
            }
            current_edge += VECTOR_LENGTH;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
