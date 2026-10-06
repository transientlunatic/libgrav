/**
 * grav.h — C interface to the Grav shared library.
 *
 * Link against:
 *   Linux/FreeBSD : libgrav.so
 *   macOS         : libgrav.dylib
 *   Windows       : grav.dll
 *
 * All masses are in kilograms.  All angles are in radians.
 * Spin magnitudes are dimensionless (0–1).
 *
 * Example (gcc):
 *   gcc example.c -L./target/release -lgrav -Wl,-rpath,./target/release -o example
 */

#ifndef GRAV_H
#define GRAV_H

#ifdef __cplusplus
extern "C" {
#endif

/** Solar mass in kilograms. */
#define GRAV_MSUN 1.988416e30

/**
 * Total mass M = m1 + m2 (kg).
 */
double grav_total_mass(double m1_kg, double m2_kg);

/**
 * Mass ratio q = m2 / m1 (dimensionless).
 * Requires m1 >= m2.
 */
double grav_mass_ratio(double m1_kg, double m2_kg);

/**
 * Symmetric mass ratio eta = m1*m2 / M^2, in (0, 0.25] (dimensionless).
 */
double grav_symmetric_mass_ratio(double m1_kg, double m2_kg);

/**
 * Chirp mass Mc = (m1*m2)^(3/5) / M^(1/5) (kg).
 */
double grav_chirp_mass(double m1_kg, double m2_kg);

/**
 * Primary mass m1 (kg) from chirp mass Mc (kg) and mass ratio q = m2/m1.
 * Requires q in (0, 1].
 */
double grav_m1_from_mc_q(double mc_kg, double q);

/**
 * Secondary mass m2 (kg) from chirp mass Mc (kg) and mass ratio q = m2/m1.
 * Requires q in (0, 1].
 */
double grav_m2_from_mc_q(double mc_kg, double q);

/**
 * Primary mass m1 (kg) from chirp mass Mc (kg) and symmetric mass ratio eta.
 * Requires eta in (0, 0.25].
 */
double grav_m1_from_mc_eta(double mc_kg, double eta);

/**
 * Secondary mass m2 (kg) from chirp mass Mc (kg) and symmetric mass ratio eta.
 * Requires eta in (0, 0.25].
 */
double grav_m2_from_mc_eta(double mc_kg, double eta);

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
double grav_chi_eff(
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
double grav_chi_p(
    double m1_kg, double m2_kg,
    double a1, double a2,
    double tilt1, double tilt2
);

/**
 * Cartesian spin components in the L-frame (dimensionless).
 * S1 = a1*(sin(tilt1), 0, cos(tilt1))
 * S2 = a2*(sin(tilt2)*cos(phi12), sin(tilt2)*sin(phi12), cos(tilt2))
 *
 * @param a1      Spin magnitude of body 1, 0-1
 * @param a2      Spin magnitude of body 2, 0-1
 * @param tilt1   Spin tilt angle of body 1 (radians, 0-pi)
 * @param tilt2   Spin tilt angle of body 2 (radians, 0-pi)
 * @param phi12   Azimuthal angle of spin 2 relative to spin 1 (radians)
 */
double grav_spin_components_s1x(double a1, double a2, double tilt1, double tilt2, double phi12);
double grav_spin_components_s1y(double a1, double a2, double tilt1, double tilt2, double phi12);
double grav_spin_components_s1z(double a1, double a2, double tilt1, double tilt2, double phi12);
double grav_spin_components_s2x(double a1, double a2, double tilt1, double tilt2, double phi12);
double grav_spin_components_s2y(double a1, double a2, double tilt1, double tilt2, double phi12);
double grav_spin_components_s2z(double a1, double a2, double tilt1, double tilt2, double phi12);

/**
 * Newtonian orbital angular momentum magnitude |L_N| (kg m^2 s^-1).
 *
 * @param m1_kg  Component mass 1 (kg)
 * @param m2_kg  Component mass 2 (kg)
 * @param f_ref  Reference GW frequency (Hz)
 */
double grav_orbital_angular_momentum(double m1_kg, double m2_kg, double f_ref);

/**
 * Precessing-spin frame transform: converts bilby/LALInference precessing
 * spin parameters to the inclination and Cartesian spin components used by
 * waveform generators.  A from-scratch reimplementation of LALSimulation's
 * SimInspiralTransformPrecessingNewInitialConditions (no LALSuite
 * dependency); matches its output to within floating-point precision.
 *
 * @param theta_jn  Inclination of J relative to the line of sight (radians)
 * @param phi_jl    Azimuth of L_N about J (radians)
 * @param tilt1     Spin tilt angle of body 1 from L_N (radians, 0-pi)
 * @param tilt2     Spin tilt angle of body 2 from L_N (radians, 0-pi)
 * @param phi12     Azimuthal angle of spin 2 relative to spin 1 (radians)
 * @param a1        Spin magnitude of body 1, 0-1
 * @param a2        Spin magnitude of body 2, 0-1
 * @param m1_kg     Component mass 1 (kg)
 * @param m2_kg     Component mass 2 (kg)
 * @param f_ref     Reference GW frequency (Hz), must be nonzero
 * @param phase     Reference orbital phase (radians)
 *
 * _iota returns the inclination of L_N relative to the line of sight
 * (radians); _s1x.._s2z return the Cartesian spin components.
 */
double grav_transform_precessing_spins_iota(
    double theta_jn, double phi_jl,
    double tilt1, double tilt2, double phi12,
    double a1, double a2,
    double m1_kg, double m2_kg,
    double f_ref, double phase
);
double grav_transform_precessing_spins_s1x(
    double theta_jn, double phi_jl,
    double tilt1, double tilt2, double phi12,
    double a1, double a2,
    double m1_kg, double m2_kg,
    double f_ref, double phase
);
double grav_transform_precessing_spins_s1y(
    double theta_jn, double phi_jl,
    double tilt1, double tilt2, double phi12,
    double a1, double a2,
    double m1_kg, double m2_kg,
    double f_ref, double phase
);
double grav_transform_precessing_spins_s1z(
    double theta_jn, double phi_jl,
    double tilt1, double tilt2, double phi12,
    double a1, double a2,
    double m1_kg, double m2_kg,
    double f_ref, double phase
);
double grav_transform_precessing_spins_s2x(
    double theta_jn, double phi_jl,
    double tilt1, double tilt2, double phi12,
    double a1, double a2,
    double m1_kg, double m2_kg,
    double f_ref, double phase
);
double grav_transform_precessing_spins_s2y(
    double theta_jn, double phi_jl,
    double tilt1, double tilt2, double phi12,
    double a1, double a2,
    double m1_kg, double m2_kg,
    double f_ref, double phase
);
double grav_transform_precessing_spins_s2z(
    double theta_jn, double phi_jl,
    double tilt1, double tilt2, double phi12,
    double a1, double a2,
    double m1_kg, double m2_kg,
    double f_ref, double phase
);

#ifdef __cplusplus
}
#endif

#endif /* GRAV_H */
