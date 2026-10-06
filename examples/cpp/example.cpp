/*
 * examples/cpp/example.cpp
 *
 * Demonstrates calling the libgrav C API from C++.
 *
 * Two usage styles are shown:
 *   1. Flat API via grav.h (original style, backward compatible)
 *   2. Namespace API via grav/binary.hpp (domain-organised, recommended)
 *
 * Build (from repo root):
 *   cargo build --release -p grav-capi
 *   g++ examples/cpp/example.cpp \
 *       -I bindings/julia/include \
 *       -L target/release -lgrav \
 *       -Wl,-rpath,$(pwd)/target/release \
 *       -o examples/cpp/example
 *   ./examples/cpp/example
 *
 * grav.h includes an `extern "C"` guard so the header is directly usable
 * from C++ without any modifications.
 */

#include <cmath>
#include <iostream>
#include <vector>
#include "grav/binary.hpp"   // provides grav::binary:: namespace

int main() {
    // ── Domain-organised API (recommended style) ──────────────────────────────
    namespace binary = grav::binary;

    double m1 = 30.0 * binary::MSUN;
    double m2 = 30.0 * binary::MSUN;

    std::cout << "=== GW150914-like binary (30+30 Msun) ===\n";
    std::cout << "Total mass      : " << binary::total_mass(m1, m2) / binary::MSUN   << " Msun\n";
    std::cout << "Mass ratio      : " << binary::mass_ratio(m1, m2)                  << "\n";
    std::cout << "Sym. mass ratio : " << binary::symmetric_mass_ratio(m1, m2)        << "\n";
    std::cout << "Chirp mass      : " << binary::chirp_mass(m1, m2) / binary::MSUN   << " Msun\n";

    // ── Inverse: recover component masses ────────────────────────────────────
    double mc  = binary::chirp_mass(m1, m2);
    double q   = binary::mass_ratio(m1, m2);
    double eta = binary::symmetric_mass_ratio(m1, m2);
    std::cout << "\n=== Inverse transforms ===\n";
    std::cout << "m1 from (Mc, q)   : " << binary::m1_from_chirp_mass_q(mc, q)    / binary::MSUN << " Msun\n";
    std::cout << "m2 from (Mc, q)   : " << binary::m2_from_chirp_mass_q(mc, q)    / binary::MSUN << " Msun\n";
    std::cout << "m1 from (Mc, eta) : " << binary::m1_from_chirp_mass_eta(mc, eta) / binary::MSUN << " Msun\n";
    std::cout << "m2 from (Mc, eta) : " << binary::m2_from_chirp_mass_eta(mc, eta) / binary::MSUN << " Msun\n";

    // ── Batch processing over a vector of masses ──────────────────────────────
    std::vector<double> masses = {10.0, 20.0, 30.0, 40.0, 50.0};  // Msun
    std::cout << "\n=== Chirp masses for equal-mass binaries ===\n";
    for (double m_sun : masses) {
        double m = m_sun * binary::MSUN;
        std::cout << "  m1=m2=" << m_sun << " Msun  ->  Mc="
                  << binary::chirp_mass(m, m) / binary::MSUN << " Msun\n";
    }

    return 0;
}
