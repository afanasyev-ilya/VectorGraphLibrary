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

struct SegmentData
{
    vector<int> src_ids;
    vector<int> dst_ids;

    int size;

    SegmentData()
    {
        size = 0;
    }

    ~SegmentData()
    {

    }

    void add_edge(int _src_id, int _dst_id)
    {
        src_ids.push_back(_src_id);
        dst_ids.push_back(_dst_id);
        size++;
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
    _T *new_data;
    MemoryAPI::allocate_host_array(&new_data, new_size);

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

struct VectorSegment
{
    int *src_ids[VECTOR_LENGTH];
    int *dst_ids[VECTOR_LENGTH];

    int current_ptrs[VECTOR_LENGTH];

    int *merged_src_ids;
    int *merged_dst_ids;
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
            MemoryAPI::allocate_host_array(&(src_ids[i]), _new_sizes[i]);
            MemoryAPI::allocate_host_array(&(dst_ids[i]), _new_sizes[i]);
        }
    }

    ~VectorSegment()
    {
        free();
    }

    void free()
    {
        MemoryAPI::free_host_array(merged_src_ids);
        MemoryAPI::free_host_array(merged_dst_ids);
        merged_src_ids = NULL;
        merged_dst_ids = NULL;
    }

    inline void add(int _src_id, int _dst_id, int _vector_index)
    {
        int pos = current_ptrs[_vector_index];

        src_ids[_vector_index][pos] = _src_id;
        dst_ids[_vector_index][pos] = _dst_id;

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

        MemoryAPI::allocate_host_array(&merged_src_ids, new_size);
        MemoryAPI::allocate_host_array(&merged_dst_ids, new_size);

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
            }
            shift += size;
        }
        merged_size = shift;

        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            MemoryAPI::free_host_array(src_ids[i]);
            MemoryAPI::free_host_array(dst_ids[i]);
        }
    }

    void thread_merge(VectorSegment &_other_segment)
    {
        int new_size = this->merged_size + _other_segment.merged_size;

        int *new_src_ids = merge_arrays(this->merged_src_ids, this->merged_size,
                                        _other_segment.merged_src_ids, _other_segment.merged_size);
        int *new_dst_ids = merge_arrays(this->merged_dst_ids, this->merged_size,
                                        _other_segment.merged_dst_ids, _other_segment.merged_size);

        this->free();
        _other_segment.free();

        this->merged_src_ids = new_src_ids;
        this->merged_dst_ids = new_dst_ids;
        this->merged_size = new_size;
        _other_segment.merged_size = 0;
    }

    void merge_to_graph(int *_graph_src_ids, int *_graph_dst_ids, long long &_merge_pos)
    {
        #pragma _NEC vector
        #pragma _NEC ivdep
        #pragma omp parallel for
        for(int i = 0; i < merged_size; i++)
        {
            _graph_src_ids[_merge_pos + i] = merged_src_ids[i];
            _graph_dst_ids[_merge_pos + i] = merged_dst_ids[i];
        }
        _merge_pos += merged_size;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct VectorData
{
    int *counts;
    int segments_count;

    VectorSegment *segments;

    VectorData(int _segments_count)
    {
        segments_count = _segments_count;
        MemoryAPI::allocate_host_array(&segments, _segments_count*_segments_count);
        MemoryAPI::allocate_host_array(&counts, segments_count*segments_count*VECTOR_LENGTH);
    }

    ~VectorData()
    {
        MemoryAPI::free_host_array(segments);
        MemoryAPI::free_host_array(counts);
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

    inline void add(int *_segment_positions, int *_src_ids, int *_dst_ids)
    {
        #pragma _NEC ivdep
        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            int seg = _segment_positions[i];
            segments[seg].add(_src_ids[i], _dst_ids[i], i);
        }
    }

    inline void vector_merge()
    {
        for(int seg = 0; seg < segments_count*segments_count; seg++)
        {
            segments[seg].vector_merge();
        }
    }

    inline void thread_merge(VectorData *_other_data)
    {
        for(int seg = 0; seg < segments_count*segments_count; seg++)
        {
            segments[seg].thread_merge(_other_data->segments[seg]);
        }
    }

    void merge_to_graph(int *_src_ids, int *_dst_ids)
    {
        long long merge_pos = 0;
        for(int seg = 0; seg < segments_count*segments_count; seg++)
        {
            segments[seg].merge_to_graph(_src_ids, _dst_ids, merge_pos);
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

#ifdef __USE_NEC_SX_AURORA__
void EdgesListGraph::preprocess_into_segmented()
{
    int segment_size_in_bytes = LLC_CACHE_SIZE/4;
    long long edges_count = this->edges_count;
    int segment_size = segment_size_in_bytes/sizeof(int);
    int segments_count = (this->vertices_count - 1) / segment_size + 1;
    cout << "segments count: " << segments_count << endl;

    if(segments_count <= 2) // TODO problem at 2 segments
        return;

    VectorData *vector_data[MAX_SX_AURORA_THREADS];
    for(int core = 0; core < MAX_SX_AURORA_THREADS; core++)
    {
        vector_data[core] = new VectorData(segments_count);
    }

    #pragma omp parallel
    {};

    Timer tm;
    tm.start();
    #pragma _NEC novector
    #pragma omp parallel num_threads(MAX_SX_AURORA_THREADS)
    {
        int tid = omp_get_thread_num();
        VectorData *local_vector_data = vector_data[tid];

        local_vector_data->reset_counts();

        #pragma omp for
        for (long long vec_start = 0; vec_start < edges_count; vec_start += VECTOR_LENGTH)
        {
            #pragma _NEC ivdep
            //#pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            #pragma _NEC gather_reorder
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                long long edge_pos = vec_start + i;

                if(edge_pos < edges_count)
                {
                    int src_id = src_ids[edge_pos];
                    int dst_id = dst_ids[edge_pos];

                    int src_segment = get_segment_num(src_id, segment_size);
                    int dst_segment = get_segment_num(dst_id, segment_size);

                    int seg = get_linear_segment_index(src_segment, dst_segment, segments_count);

                    local_vector_data->increase_count(i, seg);
                }
            }
        }
    }
    tm.end();
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_and_bandwidth_stats("Split to segments", this->edges_count, 6.0*sizeof(int));
    #endif

    tm.start();
    #pragma _NEC novector
    #pragma omp parallel num_threads(MAX_SX_AURORA_THREADS)
    {
        int tid = omp_get_thread_num();
        VectorData *local_vector_data = vector_data[tid];
        local_vector_data->resize_segments();
    }
    tm.end();
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_stats("Resize segments");
    #endif

    tm.start();
    #pragma _NEC novector
    #pragma omp parallel num_threads(MAX_SX_AURORA_THREADS)
    {
        int tid = omp_get_thread_num();
        VectorData *local_vector_data = vector_data[tid];

        #pragma omp for
        for(long long vec_start = 0; vec_start < edges_count; vec_start += VECTOR_LENGTH)
        {
            int src_ids_reg[VECTOR_LENGTH];
            int dst_ids_reg[VECTOR_LENGTH];
            int seg_reg[VECTOR_LENGTH];

            #pragma _NEC ivdep
            #pragma _NEC novob
            #pragma _NEC vector
            #pragma _NEC gather_reorder
            for (int i = 0; i < VECTOR_LENGTH; i++)
            {
                long long edge_pos = vec_start + i;

                if(edge_pos < edges_count)
                {
                    src_ids_reg[i] = src_ids[edge_pos];
                    dst_ids_reg[i] = dst_ids[edge_pos];

                    int src_segment = get_segment_num(src_ids_reg[i], segment_size);
                    int dst_segment = get_segment_num(dst_ids_reg[i], segment_size);

                    seg_reg[i] = get_linear_segment_index(src_segment, dst_segment, segments_count);
                }
            }

            local_vector_data->add(seg_reg, src_ids_reg, dst_ids_reg);
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
    tm.end();
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_stats("Parallel merge");
    #endif

    int merge_pos = 0;
    tm.start();
    vector_data[0]->merge_to_graph(src_ids, dst_ids);
    tm.end();
    #ifdef __PRINT_API_PERFORMANCE_STATS__
    tm.print_time_and_bandwidth_stats("Copy from vector segments", this->edges_count, 4.0*sizeof(int));
    #endif

    for(int core = 0; core < MAX_SX_AURORA_THREADS; core++)
    {
        delete vector_data[core];
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
void EdgesListGraph::preprocess_into_segmented()
{
    int segment_size_in_bytes = LLC_CACHE_SIZE/4;
    long long edges_count = this->edges_count;
    int segment_size = segment_size_in_bytes/sizeof(int);
    int segments_count = (this->vertices_count - 1) / segment_size + 1;
    cout << "segments count: " << segments_count << endl;

    double t3 = omp_get_wtime();
    vector<vector<int>> segmented_src_ids(segments_count*segments_count);
    vector<vector<int>> segmented_dst_ids(segments_count*segments_count);

    for(long long edge_pos = 0; edge_pos < edges_count; edge_pos++)
    {
        int src_id = src_ids[edge_pos];
        int dst_id = dst_ids[edge_pos];

        int src_segment = get_segment_num(src_id, segment_size);
        int dst_segment = get_segment_num(dst_id, segment_size);

        int seg = get_linear_segment_index(src_segment, dst_segment, segments_count);

        segmented_src_ids[seg].push_back(src_id);
        segmented_dst_ids[seg].push_back(dst_id);
    }

    long long copy_pos = 0;
    for(int seg = 0; seg < segments_count*segments_count; seg++)
    {
        MemoryAPI::copy(src_ids + copy_pos, &segmented_src_ids[seg][0], segmented_src_ids[seg].size());
        MemoryAPI::copy(dst_ids + copy_pos, &segmented_dst_ids[seg][0], segmented_dst_ids[seg].size());
        copy_pos += segmented_src_ids[seg].size();
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_MULTICORE__
void EdgesListGraph::preprocess_into_segmented()
{
    int segment_size_in_bytes = 32*1024/sizeof(float)//LLC_CACHE_SIZE/2;
    long long edges_count = this->edges_count;
    int segment_size = segment_size_in_bytes/sizeof(int);
    int segments_count = (this->vertices_count - 1) / segment_size + 1;
    cout << "segments count: " << segments_count << endl;

    int *src_ids = this->src_ids;
    int *dst_ids = this->dst_ids;
    long long *indexes = new long long[edges_count];
    int *work_buffer = new int[edges_count];
    for(long long i = 0; i < edges_count; i++)
        indexes[i] = i;
    stable_sort(indexes, indexes + edges_count,
                [src_ids, dst_ids, segment_size](long long _i1, long long _i2) {
                    int src_segment1 = get_segment_num(src_ids[_i1], segment_size);
                    int src_segment2 = get_segment_num(src_ids[_i2], segment_size);
                    if(src_segment1 != src_segment2)
                        return src_segment1 < src_segment2;
                    else
                    {
                        int dst_segment1 = get_segment_num(dst_ids[_i1], segment_size);
                        int dst_segment2 = get_segment_num(dst_ids[_i2], segment_size);
                        return dst_segment1 < dst_segment2;
                    }
                });

    #pragma _NEC ivdep
    #pragma omp parallel for
    for(long long i = 0; i < edges_count; i++)
    {
        work_buffer[i] = src_ids[indexes[i]];
    }

    #pragma _NEC ivdep
    #pragma omp parallel for
    for(long long i = 0; i < edges_count; i++)
    {
        src_ids[i] = work_buffer[i];
    }

    #pragma _NEC ivdep
    #pragma omp parallel for
    for(long long i = 0; i < edges_count; i++)
    {
        work_buffer[i] = dst_ids[indexes[i]];
    }

    #pragma _NEC ivdep
    #pragma omp parallel for
    for(long long i = 0; i < edges_count; i++)
    {
        dst_ids[i] = work_buffer[i];
    }

    delete[] indexes;
    delete[] work_buffer;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
