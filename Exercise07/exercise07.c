#include <omp.h>
#include <stdio.h>

#define NPOINTS 1000
#define MAX_ITERATIONS 2000

typedef struct {
    long long inside_count;
    double area;
    double execution_time;
    int threads_used;
} MandelbrotResult;

MandelbrotResult calculate_mandelbrot(int requested_threads) {
    const double x_min = -2.0;
    const double x_max = 0.5;
    const double y_min = 0.0;
    const double y_max = 1.125;

    long long inside_count = 0;
    int threads_used = 0;

    omp_set_num_threads(requested_threads);

    double tstart = omp_get_wtime();

    #pragma omp parallel default(none) \
        shared(threads_used) reduction(+:inside_count)
    {
        #pragma omp single
        {
            threads_used = omp_get_num_threads();
        }

        #pragma omp for schedule(static)
        for (int row = 0; row < NPOINTS; row++) {
            for (int column = 0; column < NPOINTS; column++) {

                double c_real =
                    x_min +
                    (column + 0.5) * (x_max - x_min) / NPOINTS;

                double c_imag =
                    y_min +
                    (row + 0.5) * (y_max - y_min) / NPOINTS;

                /*
                 * The lab sheet specifies the initial condition z = c.
                 */
                double z_real = c_real;
                double z_imag = c_imag;

                int iteration = 0;

                while (iteration < MAX_ITERATIONS) {
                    double magnitude_squared =
                        z_real * z_real + z_imag * z_imag;

                    if (magnitude_squared > 4.0) {
                        break;
                    }

                    double new_z_real =
                        z_real * z_real -
                        z_imag * z_imag +
                        c_real;

                    double new_z_imag =
                        2.0 * z_real * z_imag +
                        c_imag;

                    z_real = new_z_real;
                    z_imag = new_z_imag;

                    iteration++;
                }

                if (iteration == MAX_ITERATIONS) {
                    inside_count++;
                }
            }
        }
    }

    double tstop = omp_get_wtime();

    /*
     * Only the upper half is sampled. The result is multiplied by 2
     * because the Mandelbrot set is symmetric about the real axis.
     */
    double rectangle_area =
        (x_max - x_min) * (y_max - y_min);

    double area =
        2.0 * rectangle_area *
        (double)inside_count /
        (double)(NPOINTS * NPOINTS);

    MandelbrotResult result;

    result.inside_count = inside_count;
    result.area = area;
    result.execution_time = tstop - tstart;
    result.threads_used = threads_used;

    return result;
}

int main(void) {
    long long reference_count = -1;

    omp_set_dynamic(0);

    printf("Mandelbrot Set Area Estimation\n");
    printf("Grid size      : %d x %d\n", NPOINTS, NPOINTS);
    printf("Maximum iterations: %d\n\n", MAX_ITERATIONS);

    for (int threads = 1; threads <= 4; threads++) {
        MandelbrotResult result =
            calculate_mandelbrot(threads);

        if (threads == 1) {
            reference_count = result.inside_count;
        }

        printf("Requested threads : %d\n", threads);
        printf("Threads used      : %d\n", result.threads_used);
        printf("Points inside     : %lld\n", result.inside_count);
        printf("Estimated area    : %.8f\n", result.area);
        printf("Execution time    : %.6f seconds\n",
               result.execution_time);

        if (result.inside_count == reference_count) {
            printf("Verification      : Identical result\n");
        } else {
            printf("Verification      : Result mismatch\n");
        }

        printf("\n");
    }

    return 0;
}
