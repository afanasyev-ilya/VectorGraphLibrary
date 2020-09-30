#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int get_segment_num(int _vertex_id, int _segment_size)
{
    return _vertex_id / _segment_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int get_linear_segment_index(int _src_segment, int _dst_segment, int _segments_count)
{
    return _dst_segment * _segments_count + _src_segment;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
struct SegmentData
{
    vector<int> src_ids;
    vector<int> dst_ids;
    vector<_TEdgeWeight> weights;

    int size;

    SegmentData()
    {
        size = 0;
    }

    ~SegmentData()
    {

    }

    void add_edge(int _src_id, int _dst_id, _TEdgeWeight _weight)
    {
        src_ids.push_back(_src_id);
        dst_ids.push_back(_dst_id);
        weights.push_back(_weight);
        size++;
    }

    void add_edge(int _src_id, int _dst_id)
    {
        src_ids.push_back(_src_id);
        dst_ids.push_back(_dst_id);
        size++;
    }

    void merge_to_graph(int *_src_ids, int *_dst_ids, _TEdgeWeight *_weights, int &_merge_pos)
    {
        #pragma _NEC ivdep
        #pragma omp parallel for
        for(int i = 0; i < size; i++)
        {
            _src_ids[_merge_pos + i] = src_ids[i];
            _dst_ids[_merge_pos + i] = dst_ids[i];
            _weights[_merge_pos + i] = weights[i];
        }
        _merge_pos += size;
    }

    void merge_to_graph(int *_src_ids, int *_dst_ids, int &_merge_pos)
    {
        #pragma _NEC ivdep
        #pragma omp parallel for
        for(int i = 0; i < size; i++)
        {
            _src_ids[_merge_pos + i] = src_ids[i];
            _dst_ids[_merge_pos + i] = dst_ids[i];
        }
        _merge_pos += size;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void EdgesListGraph<_TVertexValue, _TEdgeWeight>::preprocess()
{
    int segment_size_in_bytes = 4*1024*1024; //TODO fix from LLC

    int segment_size = segment_size_in_bytes/sizeof(int);
    cout << "seg size: " << segment_size << endl;
    cout << "V: " << this->vertices_count << endl;

    int segments_count = (this->vertices_count - 1) / segment_size + 1;

    cout << "segments count: " << segments_count << endl;

    SegmentData<_TEdgeWeight> *segment_data = new SegmentData<_TEdgeWeight>[segments_count*segments_count];

    double t1 = omp_get_wtime();
    for(int edge_pos = 0; edge_pos < this->edges_count; edge_pos++)
    {
        int src_id = src_ids[edge_pos];
        int dst_id = dst_ids[edge_pos];

        int src_segment = get_segment_num(src_id, segment_size);
        int dst_segment = get_segment_num(dst_id, segment_size);

        int linear_segment_index = get_linear_segment_index(src_segment, dst_segment, segments_count);

        #ifdef __USE_WEIGHTED_GRAPHS__
        segment_data[linear_segment_index].add_edge(src_id, dst_id, weights[edge_pos]);
        #else
        segment_data[linear_segment_index].add_edge(src_id, dst_id);
        #endif
    }
    double t2 = omp_get_wtime();
    cout << "split time: " << t2 - t1 << " sec" << endl;

    int merge_pos = 0;
    t1 = omp_get_wtime();
    for(int cur_segment = 0; cur_segment < segments_count*segments_count; cur_segment++)
    {
        #ifdef __USE_WEIGHTED_GRAPHS__
        segment_data[cur_segment].merge_to_graph(src_ids, dst_ids, weights, merge_pos);
        #else
        segment_data[cur_segment].merge_to_graph(src_ids, dst_ids, merge_pos);
        #endif
    }
    t2 = omp_get_wtime();
    cout << "merge time: " << t2 - t1 << " sec" << endl;
    cout << "merge BW: " << 3.0*sizeof(int)*this->edges_count/((t2 - t1)*1e9) << " GB/s" << endl;

    delete[] segment_data;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
