#include "forces.h"

#include <omp.h>

static void clear_forces(int particle_count, double forces[]) {
    for (int i = 0; i < particle_count * 3; i++) {
        forces[i] = 0.0;
    }
}

ForceResult forces_serial(
    int particle_count,
    const double positions[],
    double forces[],
    double side,
    double cutoff
) {
    double potential = 0.0;
    double virial = 0.0;

    const double half_side = 0.5 * side;
    const double cutoff_squared = cutoff * cutoff;

    clear_forces(particle_count, forces);

    double start = omp_get_wtime();

    for (int i = 0; i < particle_count - 1; i++) {
        for (int j = i + 1; j < particle_count; j++) {
            double dx = positions[3 * i] - positions[3 * j];
            double dy = positions[3 * i + 1] - positions[3 * j + 1];
            double dz = positions[3 * i + 2] - positions[3 * j + 2];

            if (dx > half_side) {
                dx -= side;
            } else if (dx < -half_side) {
                dx += side;
            }

            if (dy > half_side) {
                dy -= side;
            } else if (dy < -half_side) {
                dy += side;
            }

            if (dz > half_side) {
                dz -= side;
            } else if (dz < -half_side) {
                dz += side;
            }

            double distance_squared =
                dx * dx + dy * dy + dz * dz;

            if (distance_squared > 0.0 &&
                distance_squared <= cutoff_squared) {

                double inverse_r2 = 1.0 / distance_squared;
                double inverse_r6 =
                    inverse_r2 * inverse_r2 * inverse_r2;
                double inverse_r12 = inverse_r6 * inverse_r6;

                potential +=
                    4.0 * (inverse_r12 - inverse_r6);

                double coefficient =
                    24.0 *
                    (2.0 * inverse_r12 - inverse_r6) *
                    inverse_r2;

                double fx = coefficient * dx;
                double fy = coefficient * dy;
                double fz = coefficient * dz;

                virial += dx * fx + dy * fy + dz * fz;

                forces[3 * i] += fx;
                forces[3 * i + 1] += fy;
                forces[3 * i + 2] += fz;

                forces[3 * j] -= fx;
                forces[3 * j + 1] -= fy;
                forces[3 * j + 2] -= fz;
            }
        }
    }

    ForceResult result;

    result.potential = potential;
    result.virial = virial;
    result.elapsed = omp_get_wtime() - start;

    return result;
}

ForceResult forces_parallel(
    int particle_count,
    const double positions[],
    double forces[],
    double side,
    double cutoff,
    int thread_count,
    int chunk_size
) {
    double potential = 0.0;
    double virial = 0.0;

    const double half_side = 0.5 * side;
    const double cutoff_squared = cutoff * cutoff;

    clear_forces(particle_count, forces);

    omp_set_num_threads(thread_count);
    omp_set_schedule(omp_sched_static, chunk_size);

    double start = omp_get_wtime();

    #pragma omp parallel for schedule(runtime) \
        reduction(+:potential, virial)
    for (int i = 0; i < particle_count - 1; i++) {
        for (int j = i + 1; j < particle_count; j++) {
            double dx = positions[3 * i] - positions[3 * j];
            double dy = positions[3 * i + 1] - positions[3 * j + 1];
            double dz = positions[3 * i + 2] - positions[3 * j + 2];

            if (dx > half_side) {
                dx -= side;
            } else if (dx < -half_side) {
                dx += side;
            }

            if (dy > half_side) {
                dy -= side;
            } else if (dy < -half_side) {
                dy += side;
            }

            if (dz > half_side) {
                dz -= side;
            } else if (dz < -half_side) {
                dz += side;
            }

            double distance_squared =
                dx * dx + dy * dy + dz * dz;

            if (distance_squared > 0.0 &&
                distance_squared <= cutoff_squared) {

                double inverse_r2 = 1.0 / distance_squared;
                double inverse_r6 =
                    inverse_r2 * inverse_r2 * inverse_r2;
                double inverse_r12 = inverse_r6 * inverse_r6;

                potential +=
                    4.0 * (inverse_r12 - inverse_r6);

                double coefficient =
                    24.0 *
                    (2.0 * inverse_r12 - inverse_r6) *
                    inverse_r2;

                double fx = coefficient * dx;
                double fy = coefficient * dy;
                double fz = coefficient * dz;

                virial += dx * fx + dy * fy + dz * fz;

                /*
                 * Multiple outer-loop iterations may update the same
                 * particle. The critical region prevents a data race
                 * while updating the shared force array.
                 */
                #pragma omp critical(force_update)
                {
                    forces[3 * i] += fx;
                    forces[3 * i + 1] += fy;
                    forces[3 * i + 2] += fz;

                    forces[3 * j] -= fx;
                    forces[3 * j + 1] -= fy;
                    forces[3 * j + 2] -= fz;
                }
            }
        }
    }

    ForceResult result;

    result.potential = potential;
    result.virial = virial;
    result.elapsed = omp_get_wtime() - start;

    return result;
}
