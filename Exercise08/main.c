#include "forces.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static void create_lattice(
    int points_per_side,
    double spacing,
    double positions[]
) {
    int particle = 0;

    for (int x = 0; x < points_per_side; x++) {
        for (int y = 0; y < points_per_side; y++) {
            for (int z = 0; z < points_per_side; z++) {
                positions[3 * particle] =
                    (x + 0.5) * spacing;

                positions[3 * particle + 1] =
                    (y + 0.5) * spacing;

                positions[3 * particle + 2] =
                    (z + 0.5) * spacing;

                particle++;
            }
        }
    }
}

static double maximum_force_difference(
    int particle_count,
    const double serial_forces[],
    const double parallel_forces[]
) {
    double maximum_difference = 0.0;

    for (int i = 0; i < particle_count * 3; i++) {
        double difference =
            fabs(serial_forces[i] - parallel_forces[i]);

        if (difference > maximum_difference) {
            maximum_difference = difference;
        }
    }

    return maximum_difference;
}

static void print_parallel_result(
    int thread_count,
    int chunk_size,
    ForceResult serial_result,
    ForceResult parallel_result,
    double maximum_difference
) {
    double potential_difference =
        fabs(serial_result.potential -
             parallel_result.potential);

    double virial_difference =
        fabs(serial_result.virial -
             parallel_result.virial);

    int verified =
        maximum_difference < 0.000001 &&
        potential_difference < 0.000001 &&
        virial_difference < 0.000001;

    printf("Threads              : %d\n", thread_count);
    printf("Static chunk size    : %d\n", chunk_size);
    printf("Potential energy     : %.10f\n",
           parallel_result.potential);
    printf("Virial               : %.10f\n",
           parallel_result.virial);
    printf("Execution time       : %.6f seconds\n",
           parallel_result.elapsed);

    if (parallel_result.elapsed > 0.0) {
        printf("Speedup              : %.2fx\n",
               serial_result.elapsed /
               parallel_result.elapsed);
    }

    printf("Maximum force error  : %.12e\n",
           maximum_difference);
    printf("Potential difference : %.12e\n",
           potential_difference);
    printf("Virial difference    : %.12e\n",
           virial_difference);
    printf("Verification         : %s\n\n",
           verified ? "PASS" : "FAIL");
}

int main(void) {
    const int points_per_side = 12;
    const int particle_count =
        points_per_side *
        points_per_side *
        points_per_side;

    const double spacing = 1.4;
    const double side = points_per_side * spacing;
    const double cutoff = 2.5;

    double *positions =
        malloc(3 * particle_count * sizeof(double));

    double *serial_forces =
        malloc(3 * particle_count * sizeof(double));

    double *parallel_forces =
        malloc(3 * particle_count * sizeof(double));

    if (positions == NULL ||
        serial_forces == NULL ||
        parallel_forces == NULL) {

        fprintf(stderr, "Memory allocation failed.\n");

        free(positions);
        free(serial_forces);
        free(parallel_forces);

        return 1;
    }

    omp_set_dynamic(0);

    create_lattice(
        points_per_side,
        spacing,
        positions
    );

    printf("Molecular Dynamics Force Calculation\n");
    printf("Particles            : %d\n", particle_count);
    printf("Simulation box side  : %.2f\n", side);
    printf("Cutoff distance      : %.2f\n\n", cutoff);

    ForceResult serial_result =
        forces_serial(
            particle_count,
            positions,
            serial_forces,
            side,
            cutoff
        );

    printf("Serial reference\n");
    printf("Potential energy     : %.10f\n",
           serial_result.potential);
    printf("Virial               : %.10f\n",
           serial_result.virial);
    printf("Execution time       : %.6f seconds\n\n",
           serial_result.elapsed);

    printf("Thread-count comparison using schedule(static, 1)\n\n");

    for (int threads = 1; threads <= 4; threads++) {
        ForceResult parallel_result =
            forces_parallel(
                particle_count,
                positions,
                parallel_forces,
                side,
                cutoff,
                threads,
                1
            );

        double maximum_difference =
            maximum_force_difference(
                particle_count,
                serial_forces,
                parallel_forces
            );

        print_parallel_result(
            threads,
            1,
            serial_result,
            parallel_result,
            maximum_difference
        );
    }

    const int chunk_sizes[] = {1, 4, 16, 64};
    const int chunk_count =
        sizeof(chunk_sizes) / sizeof(chunk_sizes[0]);

    printf("Chunk-size comparison using 4 threads\n\n");

    for (int index = 0; index < chunk_count; index++) {
        int chunk_size = chunk_sizes[index];

        ForceResult parallel_result =
            forces_parallel(
                particle_count,
                positions,
                parallel_forces,
                side,
                cutoff,
                4,
                chunk_size
            );

        double maximum_difference =
            maximum_force_difference(
                particle_count,
                serial_forces,
                parallel_forces
            );

        print_parallel_result(
            4,
            chunk_size,
            serial_result,
            parallel_result,
            maximum_difference
        );
    }

    free(positions);
    free(serial_forces);
    free(parallel_forces);

    return 0;
}
