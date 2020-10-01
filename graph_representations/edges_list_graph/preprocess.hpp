#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline int get_segment_num(int _vertex_id, int _segment_size)
{
    return _vertex_id / _segment_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline int get_linear_segment_index(int _src_segment, int _dst_segment, int _segments_count)
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

template <typename _TEdgeWeight>
struct VectorSegment
{
    int *src_ids[VECTOR_LENGTH];
    int *dst_ids[VECTOR_LENGTH];
    _TEdgeWeight *weights[VECTOR_LENGTH];

    int current_ptrs[VECTOR_LENGTH];
    int current_sizes[VECTOR_LENGTH];

    VectorSegment()
    {
        int initial_size = 10;
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            current_ptrs[i] = 0;
            current_sizes[i] = initial_size;
            src_ids[i] = new int[initial_size];
            dst_ids[i] = new int[initial_size];
            weights[i] = new _TEdgeWeight[initial_size];
        }
    }

    ~VectorSegment()
    {
        int initial_size = 100;
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            delete []src_ids[i];
            delete []dst_ids[i];
            delete []weights[i];
        }
    }

    inline void add(int _src_id, int _dst_id, _TEdgeWeight _weight, int _vector_index)
    {
        //int pos = current_ptrs[_vector_index];

        //if(pos < current_sizes[_vector_index])
        //{
            //src_ids[_vector_index][pos] = _src_id;
            //dst_ids[_vector_index][pos] = _dst_id;
            //weights[_vector_index][pos] = _weight;
        //}

        current_ptrs[_vector_index]++;
    }

    void check_if_resize_required()
    {

    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
struct VectorData
{
    int *counts;
    VectorSegment<_TEdgeWeight> *segments;
    int segments_count;

    VectorData(int _segments_count)
    {
        segments_count = _segments_count;
        segments = new VectorSegment<_TEdgeWeight>[_segments_count*_segments_count];

        counts = new int[segments_count*segments_count*VECTOR_LENGTH];
    }

    ~VectorData()
    {
        delete[] segments;
        delete[] counts;
    }

    inline void reset_counts()
    {
        #pragma omp for
        for(int j = 0; j < segments_count*segments_count; j++)
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
                counts[j * VECTOR_LENGTH + i] = 0; // i * segments_count*segments_count + j]
    }

    inline void increase_count(int _vector_pos, int _seg_pos)
    {
        counts[_seg_pos * VECTOR_LENGTH + _vector_pos]++;
    }

    inline long long total_count()
    {
        long long sum = 0;
        #pragma _NEC novector
        for(int j = 0; j < segments_count*segments_count; j++)
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
                sum += counts[j * VECTOR_LENGTH + i];
        return sum;
    }

    inline void add(/*int *_segment_positions, int *_src_ids, int *_dst_ids, _TEdgeWeight *_weights*/)
    {
        #pragma _NEC ivdep
        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            int seg_pos = 0;//_segment_positions[i];

            //segments[seg_pos].add(_src_ids[i], _dst_ids[i], _weights[i], i);
        }
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
    // new vector part here

    VectorData<_TEdgeWeight> *vector_data[MAX_SX_AURORA_THREADS];
    for(int core = 0; core < MAX_SX_AURORA_THREADS; core++)
    {
        vector_data[core] = new VectorData<_TEdgeWeight>(segments_count);
    }

    #pragma omp parallel
    {};

    double t1 = omp_get_wtime();
    #pragma _NEC novector
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        VectorData<_TEdgeWeight> *local_vector_data = vector_data[tid];

        long long edges_count = this->edges_count;

        local_vector_data->reset_counts();

        #pragma omp for
        for(long long vec_start = 0; vec_start < edges_count; vec_start += VECTOR_LENGTH)
        {
            #pragma _NEC ivdep
            //#pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            #pragma _NEC gather_reorder
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                long long edge_pos = vec_start + i;

                int src_id = src_ids[edge_pos];
                int dst_id = dst_ids[edge_pos];

                int src_segment = get_segment_num(src_id, segment_size);
                int dst_segment = get_segment_num(dst_id, segment_size);

                int seg = get_linear_segment_index(src_segment, dst_segment, segments_count);

                local_vector_data->increase_count(i, seg);
            }

            /*#pragma _NEC ivdep
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int seg_pos = seg_reg[i];
                local_vector_data->segments[seg_pos].current_ptrs[i]++;
                //segments[seg_pos].add(_src_ids[i], _dst_ids[i], _weights[i], i);
            }*/

            //local_vector_data->add(/*seg_reg, src_ids_reg, dst_ids_reg, weights_reg*/);
        }
    };

    double t2 = omp_get_wtime();
    cout << "split time: " << t2 - t1 << " sec" << endl;
    cout << "split BW: " << 3.0*sizeof(int)*this->edges_count/((t2 - t1)*1e9) << " GB/s" << endl;

    long long check_sum = 0;
    for(int core = 0; core < MAX_SX_AURORA_THREADS; core++)
    {
        check_sum += vector_data[core]->total_count();
    }
    cout << check_sum << " vs " << this->edges_count << endl;

    for(int core = 0; core < MAX_SX_AURORA_THREADS; core++)
    {
        delete vector_data[core];
    }

        /*SegmentData<_TEdgeWeight> *segment_data = new SegmentData<_TEdgeWeight>[segments_count*segments_count];

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

        delete[] segment_data;*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
