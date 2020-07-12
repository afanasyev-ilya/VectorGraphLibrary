#include <stdio.h>
#include <stdbool.h>
#include <asl.h>
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <functional>
#include <array>

using namespace std;

#define N 100
#define D_M 1.0
#define D_S 0.5


/*
 * NOTICE:
 * Here, error checking is omitted.
 * Practical programs must surely check errors.
 */
int main(int argc, char **argv) {
    asl_sort_t sort;

    int NX = 1024*1024;

    float *kyi = new float [NX];
    float *kyo = new float [NX];


    /* Library Initialization */
    asl_library_initialize();

    /* Sorting Preparation (ascending order) */

    asl_sort_create_s(&sort, ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
    asl_sort_preallocate(sort, NX);

    /* Input Initialization */
    printf(" ===INPUT===\n");
    printf("%5s      %16s\n", "ix", "-kyi-");
    float tmp = 0.23;

#pragma omp parallel for
    for (asl_int_t ix = 0; ix < NX; ix++) {
        const asl_int_t i = ix;
        tmp = (4.0 * tmp * (1 - tmp));
        kyi[i] = tmp;
        //printf("%5d      %16.8f\n", ix, kyi[i]);
    }
    asl_random_t rng;
    /* Generator Preparation */
    asl_random_create(&rng, ASL_RANDOMMETHOD_MT19937_64);
    asl_random_distribute_normal(rng, D_M, D_S);

    /* Generation */
    printf("[NORMAL]\n");
    printf("mu    =%7.4f\n", D_M);
    printf("sigma =%7.4f\n", D_S);

    double t1 = omp_get_wtime();
    asl_random_generate_s(rng, NX, kyi);
    double t2 = omp_get_wtime();
    cout << "rng time: " << t2 - t1 << " ms" << endl;
    cout << "rng BW: " << NX * sizeof(float)/((t2 - t1)*1e9) << " GB/s" << endl;

    /* Sorting Execution */
    printf(" ===OUTPUT===\n");
    printf("%5s      %16s\n", "ix", "-kyo-");

    t1 = omp_get_wtime();
    asl_sort_execute_s(sort, NX, kyi, ASL_NULL, kyo, ASL_NULL);
    t2 = omp_get_wtime();
    cout << "time: " << t2 - t1 << " ms" << endl;
    cout << "BW: " << 2.0*NX * sizeof(float)/((t2 - t1)*1e9) << " GB/s" << endl;

    t1 = omp_get_wtime();
    std::sort(&kyi[0], &kyi[NX]);
    t2 = omp_get_wtime();
    cout << "time std: " << t2 - t1 << " ms" << endl;
    cout << "BW std: " << 2.0*NX * sizeof(float)/((t2 - t1)*1e9) << " GB/s" << endl;

    for (asl_int_t ix = 0; ix < NX; ix++) {
        const asl_int_t i = ix;
        //printf("%5d      %16.8f\n", ix, kyo[i]);
    }

    asl_random_destroy(rng);

    /* Sorting Finalization */
    asl_sort_destroy(sort);
    /* Library Finalization */
    asl_library_finalize();

    return 0;
}