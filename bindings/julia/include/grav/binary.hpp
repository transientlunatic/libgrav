/**
 * grav/binary.hpp — C++ namespace wrapper for compact binary parameters.
 *
 * Wraps the C ABI functions from grav.h inside a C++ namespace hierarchy
 * so that callers can use the physics-domain-organised interface:
 *
 *   grav::binary::chirp_mass(m1, m2)
 *
 * Include this header instead of (or in addition to) grav.h for C++ code.
 * The underlying C symbols are unchanged; this is a zero-overhead inline
 * wrapper layer.
 *
 * Usage (g++):
 *   g++ example.cpp \
 *       -I bindings/julia/include \
 *       -L target/release -lgrav \
 *       -Wl,-rpath,$(pwd)/target/release \
 *       -o example
 */

#pragma once

#include "../grav.h"

namespace grav {

/**
 * Compact binary system parameter functions for gravitational-wave astronomy.
 *
 * All masses are in kilograms (SI).  All angles are in radians.
 * Spin magnitudes are dimensionless (0–1).
 */
namespace binary {

/** Solar mass in kilograms. */
constexpr double MSUN = GRAV_MSUN;

/** Total mass M = m1 + m2 (kg). */
inline double total_mass(double m1_kg, double m2_kg) {
    return grav_total_mass(m1_kg, m2_kg);
}

/** Mass ratio q = m2 / m1 (dimensionless).  Requires m1 >= m2. */
inline double mass_ratio(double m1_kg, double m2_kg) {
    return grav_mass_ratio(m1_kg, m2_kg);
}

/** Symmetric mass ratio η = m1·m2 / M² ∈ (0, 0.25] (dimensionless). */
inline double symmetric_mass_ratio(double m1_kg, double m2_kg) {
    return grav_symmetric_mass_ratio(m1_kg, m2_kg);
}

/** Chirp mass Mc = (m1·m2)^(3/5) / M^(1/5) (kg). */
inline double chirp_mass(double m1_kg, double m2_kg) {
    return grav_chirp_mass(m1_kg, m2_kg);
}

/** Primary mass m1 (kg) from chirp mass and mass ratio q = m2/m1. */
inline double m1_from_chirp_mass_q(double mc_kg, double q) {
    return grav_m1_from_mc_q(mc_kg, q);
}

/** Secondary mass m2 (kg) from chirp mass and mass ratio q = m2/m1. */
inline double m2_from_chirp_mass_q(double mc_kg, double q) {
    return grav_m2_from_mc_q(mc_kg, q);
}

/** Primary mass m1 (kg) from chirp mass and symmetric mass ratio η. */
inline double m1_from_chirp_mass_eta(double mc_kg, double eta) {
    return grav_m1_from_mc_eta(mc_kg, eta);
}

/** Secondary mass m2 (kg) from chirp mass and symmetric mass ratio η. */
inline double m2_from_chirp_mass_eta(double mc_kg, double eta) {
    return grav_m2_from_mc_eta(mc_kg, eta);
}

/**
 * Effective inspiral spin χ_eff ∈ [−1, 1].
 *
 * @param m1_kg, m2_kg  Component masses (kg)
 * @param a1, a2        Spin magnitudes (0–1)
 * @param tilt1, tilt2  Tilt angles (radians)
 */
inline double chi_eff(
    double m1_kg, double m2_kg,
    double a1,    double a2,
    double tilt1, double tilt2)
{
    return grav_chi_eff(m1_kg, m2_kg, a1, a2, tilt1, tilt2);
}

/**
 * Effective precession spin χ_p ∈ [0, 1].  Requires m1 >= m2.
 *
 * @param m1_kg, m2_kg  Component masses (kg, m1 >= m2)
 * @param a1, a2        Spin magnitudes (0–1)
 * @param tilt1, tilt2  Tilt angles (radians)
 */
inline double chi_p(
    double m1_kg, double m2_kg,
    double a1,    double a2,
    double tilt1, double tilt2)
{
    return grav_chi_p(m1_kg, m2_kg, a1, a2, tilt1, tilt2);
}

} // namespace binary
} // namespace grav
