// Package binary provides compact binary system parameter functions for
// gravitational-wave astronomy, wrapping the libgrav C ABI via cgo.
//
// Functions are grouped under this package to mirror the domain-organised
// structure of libgrav.binary in Python and Grav.Binary in Julia.
//
// Usage (replace example.com/your-module with your actual module path):
//
//	import "example.com/your-module/binary"
//
//	mc := binary.ChirpMass(30*binary.MSUN, 30*binary.MSUN)
//
// Build prerequisites:
//
//	cargo build --release -p grav-capi
//	# → target/release/libgrav.{so,dylib,dll}
package binary

// #cgo CFLAGS:  -I../../../bindings/julia/include
// #cgo LDFLAGS: -L../../../target/release -lgrav -Wl,-rpath,../../../target/release
// #include "grav.h"
import "C"

// MSUN is the solar mass in kilograms.
const MSUN = C.GRAV_MSUN

// TotalMass returns the total mass M = m1 + m2 in kilograms.
func TotalMass(m1Kg, m2Kg float64) float64 {
	return float64(C.grav_total_mass(C.double(m1Kg), C.double(m2Kg)))
}

// MassRatio returns the mass ratio q = m2 / m1 (dimensionless).
// Requires m1 >= m2.
func MassRatio(m1Kg, m2Kg float64) float64 {
	return float64(C.grav_mass_ratio(C.double(m1Kg), C.double(m2Kg)))
}

// SymmetricMassRatio returns η = m1·m2 / M² ∈ (0, 0.25] (dimensionless).
func SymmetricMassRatio(m1Kg, m2Kg float64) float64 {
	return float64(C.grav_symmetric_mass_ratio(C.double(m1Kg), C.double(m2Kg)))
}

// ChirpMass returns the chirp mass Mc = (m1·m2)^(3/5) / M^(1/5) in kilograms.
func ChirpMass(m1Kg, m2Kg float64) float64 {
	return float64(C.grav_chirp_mass(C.double(m1Kg), C.double(m2Kg)))
}

// M1FromChirpMassQ returns the primary mass m1 (kg) from chirp mass and
// mass ratio q = m2/m1 ∈ (0, 1].
func M1FromChirpMassQ(mcKg, q float64) float64 {
	return float64(C.grav_m1_from_mc_q(C.double(mcKg), C.double(q)))
}

// M2FromChirpMassQ returns the secondary mass m2 (kg) from chirp mass and
// mass ratio q = m2/m1 ∈ (0, 1].
func M2FromChirpMassQ(mcKg, q float64) float64 {
	return float64(C.grav_m2_from_mc_q(C.double(mcKg), C.double(q)))
}

// M1FromChirpMassEta returns the primary mass m1 (kg) from chirp mass and
// symmetric mass ratio η ∈ (0, 0.25].
func M1FromChirpMassEta(mcKg, eta float64) float64 {
	return float64(C.grav_m1_from_mc_eta(C.double(mcKg), C.double(eta)))
}

// M2FromChirpMassEta returns the secondary mass m2 (kg) from chirp mass and
// symmetric mass ratio η ∈ (0, 0.25].
func M2FromChirpMassEta(mcKg, eta float64) float64 {
	return float64(C.grav_m2_from_mc_eta(C.double(mcKg), C.double(eta)))
}

// ChiEff returns the effective inspiral spin χ_eff ∈ [−1, 1].
//
//   - a1, a2: dimensionless spin magnitudes (0–1)
//   - tilt1, tilt2: spin tilt angles in radians (0–π)
func ChiEff(m1Kg, m2Kg, a1, a2, tilt1, tilt2 float64) float64 {
	return float64(C.grav_chi_eff(
		C.double(m1Kg), C.double(m2Kg),
		C.double(a1), C.double(a2),
		C.double(tilt1), C.double(tilt2),
	))
}

// ChiP returns the effective precession spin χ_p ∈ [0, 1].
// Requires m1 >= m2.
//
//   - a1, a2: dimensionless spin magnitudes (0–1)
//   - tilt1, tilt2: spin tilt angles in radians (0–π)
func ChiP(m1Kg, m2Kg, a1, a2, tilt1, tilt2 float64) float64 {
	return float64(C.grav_chi_p(
		C.double(m1Kg), C.double(m2Kg),
		C.double(a1), C.double(a2),
		C.double(tilt1), C.double(tilt2),
	))
}
