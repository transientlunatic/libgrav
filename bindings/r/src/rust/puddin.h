/**
 * puddin.h — C interface to the Puddin shared library.
 *
 * Link against:
 *   Linux/FreeBSD : libpuddin_julia.so
 *   macOS         : libpuddin_julia.dylib
 *   Windows       : puddin_julia.dll
 *
 * All masses are in kilograms.  All angles are in radians.
 * Spin magnitudes are dimensionless (0–1).
 *
 * Example (gcc):
 *   gcc example.c -L./target/release -lpuddin_julia -Wl,-rpath,./target/release -o example
 */

#ifndef PUDDIN_H
#define PUDDIN_H

#ifdef __cplusplus
extern "C" {
#endif

/** Solar mass in kilograms. */
#define PUDDIN_MSUN 1.988416e30

/**
 * Total mass M = m1 + m2 (kg).
 */
double puddin_total_mass(double m1_kg, double m2_kg);

/**
 * Mass ratio q = m2 / m1 (dimensionless).
 * Requires m1 >= m2.
 */
double puddin_mass_ratio(double m1_kg, double m2_kg);

/**
 * Symmetric mass ratio eta = m1*m2 / M^2, in (0, 0.25] (dimensionless).
 */
double puddin_symmetric_mass_ratio(double m1_kg, double m2_kg);

/**
 * Chirp mass Mc = (m1*m2)^(3/5) / M^(1/5) (kg).
 */
double puddin_chirp_mass(double m1_kg, double m2_kg);

/**
 * Primary mass m1 (kg) from chirp mass Mc (kg) and mass ratio q = m2/m1.
 * Requires q in (0, 1].
 */
double puddin_m1_from_mc_q(double mc_kg, double q);

/**
 * Secondary mass m2 (kg) from chirp mass Mc (kg) and mass ratio q = m2/m1.
 * Requires q in (0, 1].
 */
double puddin_m2_from_mc_q(double mc_kg, double q);

/**
 * Primary mass m1 (kg) from chirp mass Mc (kg) and symmetric mass ratio eta.
 * Requires eta in (0, 0.25].
 */
double puddin_m1_from_mc_eta(double mc_kg, double eta);

/**
 * Secondary mass m2 (kg) from chirp mass Mc (kg) and symmetric mass ratio eta.
 * Requires eta in (0, 0.25].
 */
double puddin_m2_from_mc_eta(double mc_kg, double eta);

/**
 * Effective inspiral spin chi_eff in [-1, 1] (dimensionless).
 *
 * @param m1_kg   Component mass 1 (kg)
 * @param m2_kg   Component mass 2 (kg)
 * @param a1      Spin magnitude of body 1, 0-1
 * @param a2      Spin magnitude of body 2, 0-1
 * @param tilt1   Spin tilt angle of body 1 (radians, 0-pi)
 * @param tilt2   Spin tilt angle of body 2 (radians, 0-pi)
 */
double puddin_chi_eff(
    double m1_kg, double m2_kg,
    double a1, double a2,
    double tilt1, double tilt2
);

/**
 * Effective precession spin chi_p in [0, 1] (dimensionless).
 * Requires m1 >= m2.
 *
 * @param m1_kg   Primary mass (kg, m1 >= m2)
 * @param m2_kg   Secondary mass (kg)
 * @param a1      Spin magnitude of body 1, 0-1
 * @param a2      Spin magnitude of body 2, 0-1
 * @param tilt1   Spin tilt angle of body 1 (radians, 0-pi)
 * @param tilt2   Spin tilt angle of body 2 (radians, 0-pi)
 */
double puddin_chi_p(
    double m1_kg, double m2_kg,
    double a1, double a2,
    double tilt1, double tilt2
);

#ifdef __cplusplus
}
#endif

#endif /* PUDDIN_H */
