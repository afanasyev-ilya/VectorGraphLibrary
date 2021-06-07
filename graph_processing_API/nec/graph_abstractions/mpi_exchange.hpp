/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

    if(begin == end)
        return;

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

    if(vgl_library_data.get_mpi_proc_num() == 1)
        return;

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

template <typename _TGraph, typename _T>
void GraphAbstractionsNEC::exchange_vertices_array(DataExchangePolicy _policy,
                                                   _TGraph &_graph,
                                                   VerticesArray<_T> &_data)
{
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
