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
    #endif

    send_buffer = NULL;
    recv_buffer = NULL;

    bool needs_header_print = true;

    #ifdef __USE_MPI__
    if(mpi_rank != 0)
        needs_header_print = false;
    #endif

    if(needs_header_print)
    {
        cout << endl;
        cout << "/ ********************* VGL Library info ********************* /" << endl;
        #ifdef __USE_MPI__
        cout << "Using " << mpi_proc_num << " MPI processes" << endl;
        cout << "Using " << omp_get_max_threads() << " openMP threads per MPI process" << endl;
        #else
        cout << "Using " << omp_get_max_threads() << " openMP threads" << endl;
        #endif
        cout << "/ ************************************************************ /" << endl << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
