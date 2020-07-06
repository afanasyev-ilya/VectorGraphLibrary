#include <stdio.h>
#include <stdbool.h>
#include <asl.h>
#define NX 30
/*
 * NOTICE:
 * Here, error checking is omitted.
 * Practical programs must surely check errors.
 */
int main(int argc, char **argv) {
    asl_sort_t sort;
    float kyi[NX], kyo[NX];

    /* Library Initialization */
    asl_library_initialize();

    /* Sorting Preparation (ascending order) */

    asl_sort_create_s(&sort, ASL_SORTORDER_ASCENDING, ASL_SORTALGORITHM_AUTO);
    asl_sort_preallocate(sort, NX);

    /* Input Initialization */
    printf(" ===INPUT===\n");
    printf("%5s      %16s\n", "ix", "-kyi-");
    float tmp = 0.23;

    for (asl_int_t ix = 0; ix < NX; ix++) {
        const asl_int_t i = ix;
        tmp = (4.0 * tmp * (1 - tmp));
        kyi[i] = tmp;
        printf("%5d      %16.8f\n", ix, kyi[i]);
    }

    /* Sorting Execution */
    printf(" ===OUTPUT===\n");
    printf("%5s      %16s\n", "ix", "-kyo-");
    asl_sort_execute_s(sort, NX, kyi, ASL_NULL, kyo, ASL_NULL);

    for (asl_int_t ix = 0; ix < NX; ix++) {
        const asl_int_t i = ix;
        printf("%5d      %16.8f\n", ix, kyo[i]);
    }

    /* Sorting Finalization */
    asl_sort_destroy(sort);
    /* Library Finalization */
    asl_library_finalize();

    return 0;
}