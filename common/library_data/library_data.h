#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum CommunicationPolicy
{
    BROADCAST_COMMUNICATION,
    CYCLE_COMMUNICATION
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum DataExchangePolicy
{
    EXCHANGE_ALL,
    RECENT_CHANGES,
    PRIVATE_DATA
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class LibraryData
{
private:
    int mpi_rank;
    int mpi_proc_num;

    char *send_buffer;
    char *recv_buffer;
    size_t max_buffer_size;

    #ifdef __USE_MPI__
    CommunicationPolicy communication_policy;
    DataExchangePolicy data_exchange_policy;
    #endif
public:
    LibraryData(){};

    inline int get_mpi_rank() { return mpi_rank; };
    inline int get_mpi_proc_num() { return mpi_proc_num; };

    #ifdef __USE_MPI__
    void allocate_exchange_buffers(size_t max_size, size_t elem_size);
    void free_exchange_buffers();

    inline char *get_send_buffer() {return send_buffer;};
    inline char *get_recv_buffer() {return recv_buffer;};

    inline void set_communication_policy(CommunicationPolicy _communication_policy);
    inline void set_data_exchange_policy(DataExchangePolicy _data_exchange_policy);

    template <typename _T, typename MergeOp>
    void exchange_data(_T *_data, int _size, MergeOp &&_merge_op);

    template <typename _T, typename MergeOp>
    void exchange_data(_T *_new_data, _T *_old_data, int _size, MergeOp &&_merge_op);
    #endif

    void init(int argc, char **argv);
    void finalize();
};

LibraryData vgl_library_data;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "init.hpp"
#include "finalize.hpp"
#include "mpi_api.hpp"
#include "mpi_exchange.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
