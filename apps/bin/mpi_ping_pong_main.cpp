#include <mpi.h>
#include <omp.h>
#include <stdlib.h>
#include <cstdio>

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

        if(rank == 0)
            printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer );

        free(A);
    }

    MPI_Finalize();

    return 0;
}
