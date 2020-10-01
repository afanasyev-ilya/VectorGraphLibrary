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

template <typename _T>
_T* merge_arrays(_T *_first, int _first_size, _T *_second, int _second_size)
{
    int new_size = _first_size + _second_size;
    _T *new_data = new _T[new_size];

    for(int i = 0; i < _first_size; i++)
    {
        new_data[i] = _first[i];
    }
    for(int i = 0; i < _second_size; i++)
    {
        new_data[_first_size + i] = _second[i];
    }
    return new_data;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template <typename _TEdgeWeight>
struct VectorSegment
{
    int *src_ids[VECTOR_LENGTH];
    int *dst_ids[VECTOR_LENGTH];
    _TEdgeWeight *weights[VECTOR_LENGTH];

    int current_ptrs[VECTOR_LENGTH];

    int *merged_src_ids;
    int *merged_dst_ids;
    _TEdgeWeight *merged_weights;
    long long merged_size;

    VectorSegment()
    {
        #pragma _NEC ivdep
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            current_ptrs[i] = 0;
        }
    }

    void resize(int *_new_sizes)
    {
        #pragma _NEC ivdep
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            current_ptrs[i] = 0;
            src_ids[i] = new int[_new_sizes[i]];
            dst_ids[i] = new int[_new_sizes[i]];
            weights[i] = new _TEdgeWeight[_new_sizes[i]];
        }
    }

    ~VectorSegment()
    {
        free();
    }

    void free()
    {
        if(merged_src_ids != NULL)
            delete []merged_src_ids;
        if(merged_dst_ids != NULL)
            delete []merged_dst_ids;
        if(merged_weights != NULL)
            delete []merged_weights;
        merged_src_ids = NULL;
        merged_dst_ids = NULL;
        merged_weights = NULL;
    }

    inline void add(int _src_id, int _dst_id, _TEdgeWeight _weight, int _vector_index)
    {
        int pos = current_ptrs[_vector_index];

        src_ids[_vector_index][pos] = _src_id;
        dst_ids[_vector_index][pos] = _dst_id;
        weights[_vector_index][pos] = _weight;

        current_ptrs[_vector_index]++;
    }

    void vector_merge()
    {
        int new_size = 0;
        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            new_size += current_ptrs[i];
        }

        merged_src_ids = new int[new_size];
        merged_dst_ids = new int[new_size];
        merged_weights = new _TEdgeWeight[new_size];

        int shift = 0;
        #pragma _NEC novector
        for(int vec = 0; vec < VECTOR_LENGTH; vec++)
        {
            int size = current_ptrs[vec];
            #pragma _NEC ivdep
            #pragma _NEC vector
            for(int i = 0; i < size; i++)
            {
                merged_src_ids[shift + i] = src_ids[vec][i];
                merged_dst_ids[shift + i] = dst_ids[vec][i];
                merged_weights[shift + i] = weights[vec][i];
            }
            shift += size;
        }
        merged_size = shift;

        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            delete []src_ids[i];
            delete []dst_ids[i];
            delete []weights[i];
        }
    }

    void thread_merge(VectorSegment<_TEdgeWeight> &_other_segment)
    {
        int new_size = this->merged_size + _other_segment.merged_size;

        int *new_src_ids = merge_arrays(this->merged_src_ids, this->merged_size,
                                        _other_segment.merged_src_ids, _other_segment.merged_size);
        int *new_dst_ids = merge_arrays(this->merged_dst_ids, this->merged_size,
                                        _other_segment.merged_dst_ids, _other_segment.merged_size);
        _TEdgeWeight *new_weights = merge_arrays(this->merged_weights, this->merged_size,
                                                 _other_segment.merged_weights, _other_segment.merged_size);

        this->free();
        _other_segment.free();

        this->merged_src_ids = new_src_ids;
        this->merged_dst_ids = new_dst_ids;
        this->merged_weights = new_weights;
        this->merged_size = new_size;
        _other_segment.merged_size = 0;
    }

    void merge_to_graph(int *_graph_src_ids, int *_graph_dst_ids, _TEdgeWeight *_graph_weights, long long &_merge_pos)
    {
        #pragma _NEC vector
        #pragma _NEC ivdep
        #pragma omp parallel for
        for(int i = 0; i < merged_size; i++)
        {
            _graph_src_ids[_merge_pos + i] = merged_src_ids[i];
            _graph_dst_ids[_merge_pos + i] = merged_dst_ids[i];
            _graph_weights[_merge_pos + i] = merged_weights[i];
        }
        _merge_pos += merged_size;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
struct VectorData
{
    int *counts;
    int segments_count;

    VectorSegment<_TEdgeWeight> *segments;

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
        for(int seg = 0; seg < segments_count*segments_count; seg++)
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
                counts[seg * VECTOR_LENGTH + i] = 0; // i * segments_count*segments_count + j]
    }

    inline void increase_count(int _vector_pos, int _seg_pos)
    {
        counts[_seg_pos * VECTOR_LENGTH + _vector_pos]++;
    }

    inline long long total_count()
    {
        long long sum = 0;
        #pragma _NEC novector
        for(int seg = 0; seg < segments_count*segments_count; seg++)
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
                sum += counts[seg * VECTOR_LENGTH + i];
        return sum;
    }

    inline void resize_segments()
    {
        #pragma _NEC novector
        for(int seg = 0; seg < segments_count*segments_count; seg++)
        {
            int new_sizes[VECTOR_LENGTH];
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
                new_sizes[i] = counts[seg * VECTOR_LENGTH + i];

            segments[seg].resize(new_sizes);
        }
    }

    inline void add(int *_segment_positions, int *_src_ids, int *_dst_ids, _TEdgeWeight *_weights)
    {
        #pragma _NEC ivdep
        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            int seg = _segment_positions[i];
            segments[seg].add(_src_ids[i], _dst_ids[i], _weights[i], i);
        }
    }

    inline void vector_merge()
    {
        for(int seg = 0; seg < segments_count*segments_count; seg++)
        {
            segments[seg].vector_merge();
        }
    }

    inline void thread_merge(VectorData<_TEdgeWeight> *_other_data)
    {
        for(int seg = 0; seg < segments_count*segments_count; seg++)
        {
            segments[seg].thread_merge(_other_data->segments[seg]);
        }
    }

    void merge_to_graph(int *_src_ids, int *_dst_ids, _TEdgeWeight *_weights)
    {
        long long merge_pos = 0;
        for(int seg = 0; seg < segments_count*segments_count; seg++)
        {
            segments[seg].merge_to_graph(_src_ids, _dst_ids, _weights, merge_pos);
        }
    }

    void print_stats()
    {
        int sum = 0;
        for(int seg = 0; seg < segments_count*segments_count; seg++)
        {
            //cout << segments[seg].merged_size << endl;
            sum += segments[seg].merged_size;
        }
        cout << "wall edges in thread part: " << sum << endl;
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
    cout << "E: " << this->edges_count << endl;

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
    #pragma omp parallel num_threads(MAX_SX_AURORA_THREADS)
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
        }

        local_vector_data->resize_segments();

        #pragma omp for
        for(long long vec_start = 0; vec_start < edges_count; vec_start += VECTOR_LENGTH)
        {
            int src_ids_reg[VECTOR_LENGTH];
            int dst_ids_reg[VECTOR_LENGTH];
            _TEdgeWeight weights_reg[VECTOR_LENGTH];
            int seg_reg[VECTOR_LENGTH];

            #pragma _NEC ivdep
            #pragma _NEC novob
            #pragma _NEC vector
            #pragma _NEC gather_reorder
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                long long edge_pos = vec_start + i;

                src_ids_reg[i] = src_ids[edge_pos];
                dst_ids_reg[i] = dst_ids[edge_pos];
                weights_reg[i] = weights[edge_pos];

                int src_segment = get_segment_num(src_ids_reg[i], segment_size);
                int dst_segment = get_segment_num(dst_ids_reg[i], segment_size);

                seg_reg[i] = get_linear_segment_index(src_segment, dst_segment, segments_count);
            }

            local_vector_data->add(seg_reg, src_ids_reg, dst_ids_reg, weights_reg);
        }

        local_vector_data->vector_merge();

        #pragma omp barrier

        for(int step = 2; step <= MAX_SX_AURORA_THREADS; step *= 2)
        {
            #pragma omp barrier

            int shift = step / 2;
            if(tid % step == 0)
            {
                vector_data[tid]->thread_merge(vector_data[tid + shift]);
            }

            #pragma omp barrier
        }
    };
    double t2 = omp_get_wtime();
    cout << "split time: " << t2 - t1 << " sec" << endl;
    cout << "split BW: " << 6.0*sizeof(int)*this->edges_count/((t2 - t1)*1e9) << " GB/s" << endl;

    for(int core = 1; core < MAX_SX_AURORA_THREADS; core++)
    {
        delete vector_data[core];
    }

    vector_data[0]->print_stats();

    int merge_pos = 0;
    t1 = omp_get_wtime();
    vector_data[0]->merge_to_graph(src_ids, dst_ids, weights);
    t2 = omp_get_wtime();
    cout << "merge time: " << t2 - t1 << " sec" << endl;
    cout << "merge BW: " << 6.0*sizeof(int)*this->edges_count/((t2 - t1)*1e9) << " GB/s" << endl;

    delete vector_data[0];
    cout << "done" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
