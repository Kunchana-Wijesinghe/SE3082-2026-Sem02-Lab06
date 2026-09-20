#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000
#define STRIP_SIZE 1024

int main(void) {
    float *A = aligned_alloc(64, N * sizeof(float));
    float *B = aligned_alloc(64, N * sizeof(float));
    float *serial_C = aligned_alloc(64, N * sizeof(float));
    float *parallel_C = aligned_alloc(64, N * sizeof(float));

    if (A == NULL || B == NULL ||
        serial_C == NULL || parallel_C == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");

        free(A);
        free(B);
        free(serial_C);
        free(parallel_C);

        return 1;
    }

    for (int i = 0; i < N; i++) {
        A[i] = (float)(i % 1000) * 0.5f;
        B[i] = (float)(i % 500) * 0.25f;
    }

    double start = omp_get_wtime();

    for (int i = 0; i < N; i++) {
        serial_C[i] = A[i] * B[i];
    }

    double serial_time = omp_get_wtime() - start;

    start = omp_get_wtime();

    /*
     * The outer loop divides the arrays into fixed-size strips.
     * Different strips are distributed among OpenMP threads.
     */
    #pragma omp parallel for schedule(static)
    for (int strip_start = 0;
         strip_start < N;
         strip_start += STRIP_SIZE) {

        int strip_end = strip_start + STRIP_SIZE;

        if (strip_end > N) {
            strip_end = N;
        }

        /*
         * Vectorize the element-wise multiplication inside each strip.
         */
        #pragma omp simd
        for (int i = strip_start; i < strip_end; i++) {
            parallel_C[i] = A[i] * B[i];
        }
    }

    double parallel_time = omp_get_wtime() - start;

    int errors = 0;
    double checksum = 0.0;

    for (int i = 0; i < N; i++) {
        if (fabsf(serial_C[i] - parallel_C[i]) > 0.000001f) {
            errors++;
        }

        checksum += parallel_C[i];
    }

    printf("Array size              = %d\n", N);
    printf("Strip size              = %d\n", STRIP_SIZE);
    printf("Threads used            = %d\n", omp_get_max_threads());
    printf("Serial execution time   = %.6f seconds\n", serial_time);
    printf("Parallel execution time = %.6f seconds\n", parallel_time);

    if (parallel_time > 0.0) {
        printf("Speedup                 = %.2fx\n",
               serial_time / parallel_time);
    }

    printf("Verification errors     = %d\n", errors);
    printf("Checksum                = %.2f\n", checksum);

    if (errors == 0) {
        printf("Verification: Results are identical.\n");
    } else {
        printf("Verification: Results are different.\n");
    }

    free(A);
    free(B);
    free(serial_C);
    free(parallel_C);

    return 0;
}
