#ifndef FORCES_H
#define FORCES_H

typedef struct {
    double potential;
    double virial;
    double elapsed;
} ForceResult;

ForceResult forces_serial(
    int particle_count,
    const double positions[],
    double forces[],
    double side,
    double cutoff
);

ForceResult forces_parallel(
    int particle_count,
    const double positions[],
    double forces[],
    double side,
    double cutoff,
    int thread_count,
    int chunk_size
);

#endif
