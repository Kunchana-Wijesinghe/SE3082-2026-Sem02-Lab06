#include <omp.h>
#include <stdio.h>

long long fib_serial(int n) {
    if (n < 2) {
        return n;
    }

    return fib_serial(n - 1) + fib_serial(n - 2);
}

long long fib_parallel(int n, int cutoff) {
    long long i;
    long long j;

    if (n < 2) {
        return n;
    }

    /*
     * Small Fibonacci calculations are performed serially to avoid
     * creating too many small OpenMP tasks.
     */
    if (n <= cutoff) {
        return fib_serial(n);
    }

    #pragma omp task shared(i) firstprivate(n, cutoff)
    {
        i = fib_parallel(n - 1, cutoff);
    }

    #pragma omp task shared(j) firstprivate(n, cutoff)
    {
        j = fib_parallel(n - 2, cutoff);
    }

    #pragma omp taskwait

    return i + j;
}

int main(void) {
    const int n = 40;
    const int cutoff = 20;

    long long serial_result;
    long long parallel_result;

    double start;
    double stop;

    start = omp_get_wtime();
    serial_result = fib_serial(n);
    stop = omp_get_wtime();

    printf("Serial Fibonacci(%d)   = %lld\n", n, serial_result);
    printf("Serial execution time  = %.6f seconds\n", stop - start);

    start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            parallel_result = fib_parallel(n, cutoff);
        }
    }

    stop = omp_get_wtime();

    printf("Parallel Fibonacci(%d) = %lld\n", n, parallel_result);
    printf("Parallel execution time = %.6f seconds\n", stop - start);
    printf("Threads used             = %d\n", omp_get_max_threads());

    if (serial_result == parallel_result) {
        printf("Verification: Results are identical.\n");
    } else {
        printf("Verification: Results are different.\n");
    }

    return 0;
}
