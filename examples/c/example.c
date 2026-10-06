/*
 * examples/c/example.c
 *
 * Demonstrates calling the libgrav C API from C.
 *
 * Build (from repo root):
 *   cargo build --release -p grav-capi
 *   gcc examples/c/example.c \
 *       -I bindings/julia/include \
 *       -L target/release -lgrav \
 *       -Wl,-rpath,$(pwd)/target/release \
 *       -o examples/c/example -lm
 *   ./examples/c/example
 */

#include <stdio.h>
#include "grav.h"

int main(void) {
    double m1 = 30.0 * GRAV_MSUN;
    double m2 = 30.0 * GRAV_MSUN;

    printf("=== GW150914-like binary (30+30 Msun) ===\n");
    printf("Total mass         : %.2f Msun\n", grav_total_mass(m1, m2) / GRAV_MSUN);
    printf("Mass ratio         : %.4f\n",       grav_mass_ratio(m1, m2));
    printf("Sym. mass ratio    : %.4f\n",       grav_symmetric_mass_ratio(m1, m2));
    printf("Chirp mass         : %.4f Msun\n",  grav_chirp_mass(m1, m2) / GRAV_MSUN);

    /* Inverse: recover component masses from chirp mass + mass ratio */
    double mc = grav_chirp_mass(m1, m2);
    double q  = grav_mass_ratio(m1, m2);
    printf("m1 from (Mc, q)    : %.4f Msun\n", grav_m1_from_mc_q(mc, q)  / GRAV_MSUN);
    printf("m2 from (Mc, q)    : %.4f Msun\n", grav_m2_from_mc_q(mc, q)  / GRAV_MSUN);

    /* Inverse: recover component masses from chirp mass + sym. mass ratio */
    double eta = grav_symmetric_mass_ratio(m1, m2);
    printf("m1 from (Mc, eta)  : %.4f Msun\n", grav_m1_from_mc_eta(mc, eta) / GRAV_MSUN);
    printf("m2 from (Mc, eta)  : %.4f Msun\n", grav_m2_from_mc_eta(mc, eta) / GRAV_MSUN);

    /* Mild spin, 30 degrees off axis */
    double a1 = 0.3, a2 = 0.2;
    double tilt1 = 0.5236, tilt2 = 1.0472;   /* 30 deg, 60 deg in radians */
    printf("chi_eff            : %.4f\n",
           grav_chi_eff(m1, m2, a1, a2, tilt1, tilt2));
    printf("chi_p              : %.4f\n",
           grav_chi_p(m1, m2, a1, a2, tilt1, tilt2));

    return 0;
}
