/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double t_mpi_send = 0;
double t_preprocess = 0;
double t_postprocess = 0;
double t_merge = 0;
double non_opt_time = 0;

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
    int changes_count = generic_dense_copy_if(copy_cond, output_indexes, tmp_indexes_buffer, _size, 0, DONT_SAVE_ORDER);

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
void LibraryData::exchange_data_cycle_mode(_T *_new_data, int _size, MergeOp &&_merge_op, _T *_old_data,
                                           int _proc_shift)
{
    DataExchangePolicy cur_data_exchange_policy = data_exchange_policy;
    if(_old_data == NULL) // use SEND_ALL for simple exchanges (like for algorithm convergence)
        cur_data_exchange_policy = SEND_ALL;

    _T *received_data;

    size_t send_size = 0;
    size_t recv_size = 0;
    int send_elements = 0;
    int recv_elements = 0;
    char *send_ptr = NULL;
    char *recv_ptr = NULL;

    int source = (get_mpi_rank() + _proc_shift);
    int dest = (get_mpi_rank() - _proc_shift);
    if(source >= get_mpi_proc_num())
        source = 0;
    if(dest < 0)
        dest = get_mpi_proc_num() - _proc_shift;

    if(cur_data_exchange_policy == SEND_ALL)
    {
        send_elements = _size;
        recv_elements = _size;
        received_data = (_T*) recv_buffer;
        send_size = _size * sizeof(_T);
        recv_size = _size * sizeof(_T);
        send_ptr = (char *)_new_data;
        recv_ptr = (char *) received_data;
    }
    else if(cur_data_exchange_policy == RECENTLY_CHANGED)
    {
        send_elements = prepare_exchange_data(_new_data, _old_data, _size);
        recv_elements = get_recv_size(send_elements, source, dest);
        send_size = (sizeof(_T) + sizeof(int))*send_elements;
        recv_size = (sizeof(_T) + sizeof(int))*recv_elements;
        send_ptr = send_buffer;
        recv_ptr = recv_buffer;
    }

    MPI_Sendrecv(send_ptr, send_size, MPI_CHAR,
                 dest, 0, recv_ptr, recv_size, MPI_CHAR,
                 source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if(cur_data_exchange_policy == RECENTLY_CHANGED)
    {
        received_data = (_T *) send_buffer; // this is ok, we don't want to delete data either in old or recv_buffer
        #pragma _NEC ivdep
        #pragma omp parallel for
        for (int i = 0; i < _size; i++) {
            received_data[i] = _new_data[i];
        }
        parse_received_data(received_data, recv_buffer, recv_elements);
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
void LibraryData::exchange_data(_T *_new_data, int _size, MergeOp &&_merge_op, _T *_old_data)
{
    Timer tm;
    tm.start();

    if(communication_policy == CYCLE_COMMUNICATION)
    {
        for(int cur_shift = 1; cur_shift <= get_mpi_proc_num()/2; cur_shift *= 2)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            exchange_data_cycle_mode(_new_data, _size, _merge_op, _old_data, cur_shift);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    else
    {
        throw "Error: unsupported communication policy";
    }

    tm.end();
    performance_stats.update_MPI_time(tm);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
