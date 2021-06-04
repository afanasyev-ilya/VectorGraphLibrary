#include <mpi.h>
#include <omp.h>
#include <stdlib.h>
#include <cstdio>
#include <vector>
#include <iostream>

using namespace std;

extern "C" int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Status stat;

    int first = atoi(argv[1]);
    int second = atoi(argv[2]);

    vector<double> bws;

    for(int i=0; i<=26; i++)
    {
        long int N = 1 << i;

        // Allocate memory for A on CPU
        double *A = (double *) malloc(N * sizeof(double));

        // Initialize all elements of A to 0.0
        for(int i=0; i<N; i++)
        {
            A[i] = 0.0;
        }

        int tag1 = 10;
        int tag2 = 20;

        int loop_count = 50;

        // Warm-up loop
        for(int i=1; i<=5; i++){
            if(rank == first){
                MPI_Send(A, N, MPI_DOUBLE, second, tag1, MPI_COMM_WORLD);
                MPI_Recv(A, N, MPI_DOUBLE, second, tag2, MPI_COMM_WORLD, &stat);
            }
            else if(rank == second){
                MPI_Recv(A, N, MPI_DOUBLE, first, tag1, MPI_COMM_WORLD, &stat);
                MPI_Send(A, N, MPI_DOUBLE, first, tag2, MPI_COMM_WORLD);
            }
        }

        // Time ping-pong for loop_count iterations of data transfer size 8*N bytes
        double start_time, stop_time, elapsed_time;
        start_time = MPI_Wtime();

        for(int i=1; i<=loop_count; i++){
            if(rank == first){
                MPI_Send(A, N, MPI_DOUBLE, second, tag1, MPI_COMM_WORLD);
                MPI_Recv(A, N, MPI_DOUBLE, second, tag2, MPI_COMM_WORLD, &stat);
            }
            else if(rank == second){
                MPI_Recv(A, N, MPI_DOUBLE, first, tag1, MPI_COMM_WORLD, &stat);
                MPI_Send(A, N, MPI_DOUBLE, first, tag2, MPI_COMM_WORLD);
            }
        }

        stop_time = MPI_Wtime();
        elapsed_time = stop_time - start_time;

        long int num_B = 8*N;
        long int B_in_GB = 1 << 30;
        double num_GB = (double)num_B / (double)B_in_GB;
        double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);

        bws.push_back(num_GB / avg_time_per_transfer);

        if(rank == first)
        {
            printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n", num_B,
                   avg_time_per_transfer, num_GB / avg_time_per_transfer);
        }
        free(A);
    }

    if(rank == first)
    {
        for(int i = 0; i < bws.size(); i++)
        {
            cout << bws[i] << endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    bws.clear();

    for(int i=0; i<=26; i++)
    {
        long int N = 1 << i;
        double *A = (double *) malloc(N * sizeof(double));
        long int per_proc_size = N / size;

        // Initialize all elements of A to 0.0
        for(int i = 0; i < N; i++)
        {
            A[i] = 0.0;
        }

        MPI_Allgather(&A[per_proc_size*rank], per_proc_size, MPI_DOUBLE, A, per_proc_size,
                      MPI_DOUBLE, MPI_COMM_WORLD);

        int loop_count = 50;

        double start_time, stop_time, elapsed_time;
        start_time = MPI_Wtime();

        for(int i=1; i<=loop_count; i++){
            MPI_Allgather(&A[per_proc_size*rank], per_proc_size, MPI_DOUBLE, A, per_proc_size,
                          MPI_DOUBLE, MPI_COMM_WORLD);
        }

        stop_time = MPI_Wtime();
        elapsed_time = stop_time - start_time;

        long int num_B = sizeof(double)*N;
        long int B_in_GB = 1 << 30;
        double num_GB = (double)num_B / (double)B_in_GB;
        double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);
        elapsed_time = stop_time - start_time;
        bws.push_back(num_GB / avg_time_per_transfer);

        if(rank == first)
        {
            printf("Allgather size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n", num_B,
                   avg_time_per_transfer, num_GB / avg_time_per_transfer);
        }
    }

    if(rank == first)
    {
        for(int i = 0; i < bws.size(); i++)
        {
            cout << bws[i] << endl;
        }
    }

    MPI_Finalize();

    return 0;
}
