#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void LibraryData::init(int argc, char **argv)
{
    #ifdef __USE_MPI__
    mpi_proc_num = 0;
    mpi_rank = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_proc_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    communication_policy = CYCLE_COMMUNICATION;
    data_exchange_policy = SEND_ALL;
    #endif

    send_buffer = NULL;
    recv_buffer = NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
