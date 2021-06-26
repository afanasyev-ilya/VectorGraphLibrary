/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline int get_recv_size(int _send_size, int _source, int _dest)
{
    int recv_size = 0;
    MPI_Sendrecv(&_send_size, 1, MPI_INT,
                 _dest, 0, &recv_size, 1, MPI_INT,
                 _source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return recv_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
inline int prepare_exchange_data(_T *_new, _T *_old, int _size)
{
    char *send_buffer = vgl_library_data.get_send_buffer();
    int *output_indexes = (int*)send_buffer;

    char *recv_buffer = vgl_library_data.get_recv_buffer();
    int *tmp_indexes_buffer = (int*)recv_buffer;

    auto copy_cond = [&_new, &_old](int i)->float
    {
        int result = -1;
        if(_new[i] != _old[i])
            result = 1;
        return result;
    };
    //double t1 = omp_get_wtime();
    int changes_count = generic_dense_copy_if(copy_cond, output_indexes, tmp_indexes_buffer, _size, 0, DONT_SAVE_ORDER);
    //double t2 = omp_get_wtime();
    //cout << "copy if time: " << (t2 - t1) *1000.0 << " ms " << vgl_library_data.get_mpi_rank() << endl;
    //cout << "copy if BW: " << _size * 2.0 * sizeof(int) / ((t2 - t1)*1e9) << " GB/s " << vgl_library_data.get_mpi_rank() << endl;

    _T *output_data = (_T*) (&send_buffer[changes_count*sizeof(int)]);
    _T *tmp_data_buffer = (_T*) (&recv_buffer[changes_count*sizeof(int)]);

    #pragma _NEC cncall
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma _NEC sparse
    #pragma _NEC gather_reorder
    #pragma omp parallel for
    for(int i = 0; i < changes_count; i++)
    {
        output_data[i] = _new[output_indexes[i]];
    }

    return changes_count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
inline void parse_received_data(_T *_data, char *_buffer, int _recv_size)
{
    int *index_buffer = (int*) _buffer;
    _T *data_buffer = (_T*) (&_buffer[_recv_size*sizeof(int)]);

    #pragma _NEC cncall
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma _NEC gather_reorder
    #pragma omp parallel for
    for(int i = 0; i < _recv_size; i++)
    {
        _data[index_buffer[i]] = data_buffer[i];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename MergeOp>
void exchange_data_cycle_mode(_T *_new_data, int _size, MergeOp &&_merge_op, _T *_old_data,
                              int _proc_shift)
{
    DataExchangePolicy cur_data_exchange_policy = EXCHANGE_RECENTLY_CHANGED;
    if(_old_data == NULL) // use EXCHANGE_ALL for simple exchanges (like for algorithm convergence)
        cur_data_exchange_policy = EXCHANGE_ALL;

    _T *received_data;

    size_t send_size = 0;
    size_t recv_size = 0;
    int send_elements = 0;
    int recv_elements = 0;
    char *send_ptr = NULL;
    char *recv_ptr = NULL;

    int source = (vgl_library_data.get_mpi_rank() + _proc_shift) % vgl_library_data.get_mpi_proc_num();
    int dest = (vgl_library_data.get_mpi_rank() - _proc_shift);
    if(dest < 0)
        dest = vgl_library_data.get_mpi_proc_num() + dest;

    if(cur_data_exchange_policy == EXCHANGE_ALL)
    {
        send_elements = _size;
        recv_elements = _size;
        received_data = (_T*) vgl_library_data.get_recv_buffer();
        send_size = _size * sizeof(_T);
        recv_size = _size * sizeof(_T);
        send_ptr = (char *)_new_data;
        recv_ptr = (char *) received_data;
    }
    else if(cur_data_exchange_policy == EXCHANGE_RECENTLY_CHANGED)
    {
        send_elements = prepare_exchange_data(_new_data, _old_data, _size);
        recv_elements = get_recv_size(send_elements, source, dest);
        send_size = (sizeof(_T) + sizeof(int))*send_elements;
        recv_size = (sizeof(_T) + sizeof(int))*recv_elements;
        send_ptr = vgl_library_data.get_send_buffer();
        recv_ptr = vgl_library_data.get_recv_buffer();
    }

    Timer send_recv_tm;
    send_recv_tm.start();
    MPI_Sendrecv(send_ptr, send_size, MPI_CHAR,
                 dest, 0, recv_ptr, recv_size, MPI_CHAR,
                 source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    send_recv_tm.end();
    performance_stats.update_MPI_functions_time(send_recv_tm);

    if(cur_data_exchange_policy == EXCHANGE_RECENTLY_CHANGED)
    {
        received_data = (_T *) vgl_library_data.get_send_buffer(); // this is ok, we don't want to delete data either in old or recv_buffer
        #pragma _NEC ivdep
        #pragma omp parallel for
        for (int i = 0; i < _size; i++) {
            received_data[i] = _new_data[i];
        }
        parse_received_data(received_data, vgl_library_data.get_recv_buffer(), recv_elements);
    }
    #ifdef __USE_NEC_SX_AURORA__
    #pragma _NEC cncall
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma _NEC gather_reorder
    #endif
    #pragma omp parallel for
    for(int i = 0; i < _size; i++)
    {
        _new_data[i] = _merge_op(received_data[i], _new_data[i]);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename MergeOp>
void exchange_data_recently_changed_and_all(_T *_new_data, int _size, MergeOp &&_merge_op, _T *_old_data)
{
    MPI_Barrier(MPI_COMM_WORLD);
    Timer tm;
    tm.start();

    int mpi_proc_num = vgl_library_data.get_mpi_proc_num();

    if((mpi_proc_num & (mpi_proc_num - 1)) == 0) // is power of 2
    {
        int cur_shift = 1;
        for(int cur_shift = 1; cur_shift <= mpi_proc_num/2; cur_shift *= 2)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            exchange_data_cycle_mode(_new_data, _size, _merge_op, _old_data, cur_shift);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    else
    {
        for(int i = 0; i < mpi_proc_num; i++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            exchange_data_cycle_mode(_new_data, _size, _merge_op, _old_data, 1);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    tm.end();
    performance_stats.update_MPI_time(tm);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename _T>
void templated_allgatherv(_T *_data, int *_recv_shifts, int *_recv_sizes)
{
    throw "Error: unsupported datatype in templated MPI_Allgatherv";
}

template<> void templated_allgatherv<double>(double *_data, int *_recv_shifts, int *_recv_sizes)
{
    MPI_Allgatherv((&_data[_recv_shifts[vgl_library_data.get_mpi_rank()]]),
                   _recv_sizes[vgl_library_data.get_mpi_rank()], MPI_DOUBLE,
                   _data, _recv_sizes, _recv_shifts, MPI_DOUBLE, MPI_COMM_WORLD);
}

template<> void templated_allgatherv<float>(float *_data, int *_recv_shifts, int *_recv_sizes)
{
    MPI_Allgatherv((&_data[_recv_shifts[vgl_library_data.get_mpi_rank()]]),
                   _recv_sizes[vgl_library_data.get_mpi_rank()], MPI_FLOAT,
                   _data, _recv_sizes, _recv_shifts, MPI_FLOAT, MPI_COMM_WORLD);
}

template<> void templated_allgatherv<int>(int *_data, int *_recv_shifts, int *_recv_sizes)
{
    MPI_Allgatherv((&_data[_recv_shifts[vgl_library_data.get_mpi_rank()]]),
                   _recv_sizes[vgl_library_data.get_mpi_rank()], MPI_INT,
                   _data, _recv_sizes, _recv_shifts, MPI_INT, MPI_COMM_WORLD);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void in_group_exchange(_T *_data, int _begin, int _end)
{
    int begin = _begin;
    int end = _end;

    int send_size = end - begin;
    int send_shift = begin;

    const int mpi_processes = vgl_library_data.get_mpi_proc_num();
    int recv_sizes[mpi_processes];
    int recv_shifts[mpi_processes];

    recv_sizes[vgl_library_data.get_mpi_rank()] = send_size;
    recv_shifts[vgl_library_data.get_mpi_rank()] = send_shift;

    Timer comm_tm;
    comm_tm.start();
    MPI_Allgather(&send_size, 1, MPI_INT, recv_sizes, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&send_shift, 1, MPI_INT, recv_shifts, 1, MPI_INT, MPI_COMM_WORLD);

    templated_allgatherv(_data, recv_shifts, recv_sizes);
    comm_tm.end();
    performance_stats.update_MPI_functions_time(comm_tm);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void exchange_data_private(VectCSRGraph &_graph, _T *_data, int _size, TraversalDirection _direction)
{
    MPI_Barrier(MPI_COMM_WORLD);
    Timer tm;
    tm.start();

    pair<int,int> ve_part = _graph.get_direction_graph_ptr(_direction)->get_vector_engine_mpi_thresholds();
    pair<int,int> vc_part = _graph.get_direction_graph_ptr(_direction)->get_vector_core_mpi_thresholds();
    pair<int,int> coll_part = _graph.get_direction_graph_ptr(_direction)->get_collective_mpi_thresholds();

    _T *received_data = (_T*) vgl_library_data.get_recv_buffer();
    MemoryAPI::set(received_data, (_T)0, _size);

    in_group_exchange(_data,  ve_part.first,  ve_part.second);
    in_group_exchange(_data,  vc_part.first,  vc_part.second);
    in_group_exchange(_data,  coll_part.first,  coll_part.second);

    MPI_Barrier(MPI_COMM_WORLD);
    tm.end();
    performance_stats.update_MPI_time(tm);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TGraph, typename _T, typename MergeOp>
void GraphAbstractionsNEC::exchange_vertices_array(DataExchangePolicy _policy,
                                                   _TGraph &_graph,
                                                   VerticesArray<_T> &_data,
                                                   VerticesArray<_T> &_old_data,
                                                   MergeOp &&_merge_op)
{
    if(vgl_library_data.get_mpi_proc_num() == 1)
        return;

    if(_policy == EXCHANGE_RECENTLY_CHANGED)
    {
        exchange_data_recently_changed_and_all(_data.get_ptr(), _data.size(), _merge_op, _old_data.get_ptr());
    }
    else
    {
        throw "Error in GraphAbstractionsNEC::exchange_vertices_array : old data is provided for NON EXCHANGE_RECENTLY_CHANGED";
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TGraph, typename _T, typename MergeOp>
void GraphAbstractionsNEC::exchange_vertices_array(DataExchangePolicy _policy,
                                                   _TGraph &_graph,
                                                   VerticesArray<_T> &_data,
                                                   MergeOp &&_merge_op)
{
    if(vgl_library_data.get_mpi_proc_num() == 1)
        return;

    if(_policy == EXCHANGE_ALL)
    {
        exchange_data_recently_changed_and_all(_data.get_ptr(), _data.size(), _merge_op, (_T*)NULL);
    }
    else
    {
        throw "Error in GraphAbstractionsNEC::exchange_vertices_array : old data is NOT provided for NON EXCHANGE_RECENTLY_CHANGED";
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TGraph, typename _T>
void GraphAbstractionsNEC::exchange_vertices_array(DataExchangePolicy _policy,
                                                   _TGraph &_graph,
                                                   VerticesArray<_T> &_data)
{
    if(vgl_library_data.get_mpi_proc_num() == 1)
        return;

    if(_policy == EXCHANGE_RECENTLY_CHANGED)
    {
         throw "Error in GraphAbstractionsNEC::exchange_vertices_array : old data must be provided for EXCHANGE_RECENTLY_CHANGED";
    }
    else if(_policy == EXCHANGE_PRIVATE_DATA)
    {
        exchange_data_private(_graph, _data.get_ptr(), _data.size(), current_traversal_direction);
    }
    else
    {
        throw "Currently not supported";
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
