/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_MPI__
void LibraryData::set_communication_policy(CommunicationPolicy _communication_policy)
{
    communication_policy = _communication_policy;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_MPI__
void LibraryData::set_data_exchange_policy(DataExchangePolicy _data_exchange_policy)
{
    data_exchange_policy = _data_exchange_policy;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_MPI__
void LibraryData::allocate_exchange_buffers(size_t _max_size, size_t _elem_size)
{
    MemoryAPI::allocate_array(&send_buffer, _max_size*_elem_size + _max_size*sizeof(int));
    MemoryAPI::allocate_array(&recv_buffer, _max_size*_elem_size + _max_size*sizeof(int));

    // heat send/recv
    int source = (this->get_mpi_rank() + 1);
    int dest = (this->get_mpi_rank() - 1);
    if(source >= this->get_mpi_proc_num())
        source = 0;
    if(dest < 0)
        dest = this->get_mpi_proc_num() - 1;
    MPI_Sendrecv(send_buffer, _max_size*_elem_size, MPI_CHAR,
                 dest, 0, recv_buffer, _max_size*_elem_size, MPI_CHAR,
                 source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int *send_buffer_i = (int*)send_buffer;
    int *recv_buffer_i = (int*)recv_buffer;
    #pragma omp parallel for
    for(size_t i = 0; i < _max_size; i++)
    {
        send_buffer_i[i] = 0;
        recv_buffer_i[i] = 0;
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_MPI__
void LibraryData::free_exchange_buffers()
{
    if(send_buffer != NULL)
        MemoryAPI::free_array(send_buffer);
    if(recv_buffer != NULL)
        MemoryAPI::free_array(recv_buffer);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
