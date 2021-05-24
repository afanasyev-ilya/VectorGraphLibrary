/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double t_small = 0;
double t_preprocess = 0;
double t_postprocess = 0;
double t_merge = 0;
double non_opt_time = 0;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
inline int estimate_changes_count(_T *_new, _T *_old, int _size)
{
    int changes_count = 0;
    #pragma omp parallel for reduction(+: changes_count)
    for(int i = 0; i < _size; i++)
    {
        if(_new[i] != _old[i])
            changes_count++;
    }
    return changes_count;
}

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
inline int test_test(const _T *_new_data, const _T *_old_data, int _size,
                     int *_out_data,
                     int *_tmp_buffer,
                     const int _buffer_size,
                     const int _threads_count = MAX_SX_AURORA_THREADS)
{
    int elements_per_thread = (_buffer_size - 1)/_threads_count + 1;
    int elements_per_vector = (elements_per_thread - 1)/VECTOR_LENGTH + 1;
    int shifts_array[MAX_SX_AURORA_THREADS];

    int elements_count = 0;
    #pragma omp parallel num_threads(_threads_count) shared(elements_count)
    {
        int tid = omp_get_thread_num();
        int start_pointers_reg[VECTOR_LENGTH];
        int current_pointers_reg[VECTOR_LENGTH];
        int last_pointers_reg[VECTOR_LENGTH];

        #pragma _NEC vreg(start_pointers_reg)
        #pragma _NEC vreg(current_pointers_reg)
        #pragma _NEC vreg(last_pointers_reg)

        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            start_pointers_reg[i] = tid * elements_per_thread + i * elements_per_vector;
            current_pointers_reg[i] = tid * elements_per_thread + i * elements_per_vector;
            last_pointers_reg[i] = tid * elements_per_thread + i * elements_per_vector;
        }

        #pragma omp for schedule(static)
        for(int vec_start = 0; vec_start < _size; vec_start += VECTOR_LENGTH)
        {
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = vec_start + i;
                if((src_id < _size) && (_new_data[src_id] != _old_data[src_id]))
                {
                    _tmp_buffer[current_pointers_reg[i]] = src_id;
                    current_pointers_reg[i]++;
                }
            }
        }

        int max_difference = 0;
        int save_values_per_thread = 0;
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            int difference = current_pointers_reg[i] - start_pointers_reg[i];
            save_values_per_thread += difference;
            if(difference > max_difference)
                max_difference = difference;
        }

        shifts_array[tid] = save_values_per_thread;
        #pragma omp barrier

        #pragma omp master
        {
            int cur_shift = 0;
            for(int i = 1; i < _threads_count; i++)
            {
                shifts_array[i] += shifts_array[i - 1];
            }

            elements_count = shifts_array[_threads_count - 1];

            for(int i = (_threads_count - 1); i >= 1; i--)
            {
                shifts_array[i] = shifts_array[i - 1];
            }
            shifts_array[0] = 0;
        }

        #pragma omp barrier

        int tid_shift = shifts_array[tid];
        int *private_ptr = &(_out_data[tid_shift]);

        int local_pos = 0;
        #pragma _NEC novector
        for(int pos = 0; pos < max_difference; pos++)
        {
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int loc_size = current_pointers_reg[i] - start_pointers_reg[i];

                if(pos < loc_size)
                {
                    private_ptr[local_pos] = _tmp_buffer[last_pointers_reg[i]];
                    last_pointers_reg[i]++;
                    local_pos++;
                }
            }
        }
    }

    return elements_count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
inline int prepare_exchange_data(_T *_new, _T *_old, int _size)
{
    int changes_count = 0;
    #pragma omp parallel for reduction(+: changes_count)
    for(int i = 0; i < _size; i++)
    {
        if(_new[i] != _old[i])
        {
            changes_count++;
        }
    }

    char *send_buffer = vgl_library_data.get_send_buffer();
    _T *output_data = (_T*) send_buffer;
    int *output_indexes = (int*)(&send_buffer[changes_count*sizeof(_T)]);

    char *recv_buffer = vgl_library_data.get_recv_buffer();
    _T *tmp_data_buffer = (_T*) recv_buffer;
    int *tmp_indexes_buffer = (int*)(&recv_buffer[changes_count*sizeof(_T)]);

    double t1 = omp_get_wtime();

    int count = test_test(_new, _old, _size, output_indexes, tmp_indexes_buffer, _size);

    double t2 = omp_get_wtime();
    cout << "copy if BW: " << _size * sizeof(_T) * 3.0 / ((t2 - t1)*1e9) << " GB/s" << endl;
    non_opt_time += t2 - t1;

    #pragma _NEC cncall
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    #pragma _NEC vector
    #pragma _NEC sparse
    #pragma _NEC gather_reorder
    for(int i = 0; i < count; i++)
    {
        output_data[i] = _new[output_indexes[i]];
    }

    return changes_count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
inline void parse_received_data(_T *_data, char *_buffer, int _recv_size)
{
    _T *data_buffer = (_T*) _buffer;
    int *index_buffer = (int*)(&_buffer[_recv_size*sizeof(_T)]);

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
void LibraryData::exchange_data(_T *_new_data, _T *_old_data, int _size, MergeOp &&_merge_op)
{
    double t1, t2;

    int source = (vgl_library_data.get_mpi_rank() + 1);
    int dest = (vgl_library_data.get_mpi_rank() - 1);
    if(source >= vgl_library_data.get_mpi_proc_num())
        source = 0;
    if(dest < 0)
        dest = vgl_library_data.get_mpi_proc_num() - 1;

    t1 = omp_get_wtime();
    int send_elements = prepare_exchange_data(_new_data, _old_data, _size);
    int recv_elements = get_recv_size(send_elements, source, dest);
    size_t send_size = (sizeof(_T) + sizeof(int))*send_elements;
    size_t recv_size = (sizeof(_T) + sizeof(int))*recv_elements;
    t2 = omp_get_wtime();
    t_preprocess += t2 - t1;

    t1 = omp_get_wtime();
    MPI_Sendrecv(get_send_buffer(), send_size, MPI_CHAR,
                 dest, 0, get_recv_buffer(), recv_size, MPI_CHAR,
                 source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    t2 = omp_get_wtime();
    cout << "SMALL send time: " << (t2 - t1)*1000 << " ms" << endl;
    cout << (recv_size + send_size)/((t2 - t1)*1e9) << " GB/s" << endl;
    t_small += t2 - t1;

    t1 = omp_get_wtime();
    _T *buffer_for_received_data = (_T*) _send_buffer; // this is ok, we don't want to delete data either in old or recv_buffer
    #pragma _NEC ivdep
    #pragma omp parallel for
    for(int i = 0; i < _size; i++)
    {
        _buffer_for_received_data[i] = _new_data[i];
    }
    parse_received_data(_buffer_for_received_data, get_recv_buffer(), recv_elements);
    t2 = omp_get_wtime();
    t_postprocess += t2 - t1;

    t1 = omp_get_wtime();
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
        _new_data[i] = _merge_op(_buffer_for_received_data[i], _new_data[i]);
    }
    t2 = omp_get_wtime();
    t_merge += t2 - t1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T, typename MergeOp>
void LibraryData::exchange_data(_T *_data, int _size, MergeOp &&_merge_op)
{
    _T *received_data = (_T*) recv_buffer;

    if(communication_policy == CYCLE_COMMUNICATION)
    {
        int source = (get_mpi_rank() + 1);
        int dest = (get_mpi_rank() - 1);
        if(source >= get_mpi_proc_num())
            source = 0;
        if(dest < 0)
            dest = get_mpi_proc_num() - 1;

        // TODO type
        MPI_Sendrecv(_data, _size, MPI_FLOAT,
                     dest, 0, received_data, _size, MPI_FLOAT,
                     source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else
    {
        throw "Error: unsupported communication policy";
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
        _data[i] = _merge_op(received_data[i], _data[i]);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
