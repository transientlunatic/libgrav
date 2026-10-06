// examples/go/example.go
//
// Demonstrates calling the libgrav binary parameter functions from Go via
// direct cgo bindings, using a local `binary` helper struct that mirrors
// the domain-organised APIs used in other languages.
//
// Build (from repo root):
//
//	cargo build --release -p grav-capi
//	cd examples/go && go run example.go
//
// Or build a binary:
//
//	cd examples/go && go build -o example && ./example

package main

// #cgo CFLAGS:  -I../../bindings/julia/include
// #cgo LDFLAGS: -L../../target/release -lgrav -Wl,-rpath,../../target/release
// #include "grav.h"
import "C"

import "fmt"

// binary wraps the C ABI under an idiomatic Go namespace, mirroring the
// libgrav.binary submodule in Python and Grav.Binary in Julia.
var binary = struct {
	MSUN             float64
	TotalMass        func(float64, float64) float64
	MassRatio        func(float64, float64) float64
	SymmetricMassRatio func(float64, float64) float64
	ChirpMass        func(float64, float64) float64
	M1FromChirpMassQ func(float64, float64) float64
	M2FromChirpMassQ func(float64, float64) float64
	M1FromChirpMassEta func(float64, float64) float64
	M2FromChirpMassEta func(float64, float64) float64
	ChiEff           func(float64, float64, float64, float64, float64, float64) float64
	ChiP             func(float64, float64, float64, float64, float64, float64) float64
}{
	MSUN: float64(C.GRAV_MSUN),
	TotalMass: func(m1, m2 float64) float64 {
		return float64(C.grav_total_mass(C.double(m1), C.double(m2)))
	},
	MassRatio: func(m1, m2 float64) float64 {
		return float64(C.grav_mass_ratio(C.double(m1), C.double(m2)))
	},
	SymmetricMassRatio: func(m1, m2 float64) float64 {
		return float64(C.grav_symmetric_mass_ratio(C.double(m1), C.double(m2)))
	},
	ChirpMass: func(m1, m2 float64) float64 {
		return float64(C.grav_chirp_mass(C.double(m1), C.double(m2)))
	},
	M1FromChirpMassQ: func(mc, q float64) float64 {
		return float64(C.grav_m1_from_mc_q(C.double(mc), C.double(q)))
	},
	M2FromChirpMassQ: func(mc, q float64) float64 {
		return float64(C.grav_m2_from_mc_q(C.double(mc), C.double(q)))
	},
	M1FromChirpMassEta: func(mc, eta float64) float64 {
		return float64(C.grav_m1_from_mc_eta(C.double(mc), C.double(eta)))
	},
	M2FromChirpMassEta: func(mc, eta float64) float64 {
		return float64(C.grav_m2_from_mc_eta(C.double(mc), C.double(eta)))
	},
	ChiEff: func(m1, m2, a1, a2, tilt1, tilt2 float64) float64 {
		return float64(C.grav_chi_eff(
			C.double(m1), C.double(m2),
			C.double(a1), C.double(a2),
			C.double(tilt1), C.double(tilt2),
		))
	},
	ChiP: func(m1, m2, a1, a2, tilt1, tilt2 float64) float64 {
		return float64(C.grav_chi_p(
			C.double(m1), C.double(m2),
			C.double(a1), C.double(a2),
			C.double(tilt1), C.double(tilt2),
		))
	},
}

func main() {
	m1 := 30.0 * binary.MSUN
	m2 := 30.0 * binary.MSUN

	fmt.Println("=== GW150914-like binary (30+30 Msun) ===")
	fmt.Printf("Total mass      : %.4f Msun\n", binary.TotalMass(m1, m2)/binary.MSUN)
	fmt.Printf("Mass ratio      : %.4f\n", binary.MassRatio(m1, m2))
	fmt.Printf("Sym. mass ratio : %.4f\n", binary.SymmetricMassRatio(m1, m2))
	fmt.Printf("Chirp mass      : %.4f Msun\n", binary.ChirpMass(m1, m2)/binary.MSUN)

	// Inverse transforms
	mc := binary.ChirpMass(m1, m2)
	q := binary.MassRatio(m1, m2)
	eta := binary.SymmetricMassRatio(m1, m2)
	fmt.Println("\n=== Inverse transforms ===")
	fmt.Printf("m1 from (Mc, q)   : %.4f Msun\n", binary.M1FromChirpMassQ(mc, q)/binary.MSUN)
	fmt.Printf("m2 from (Mc, q)   : %.4f Msun\n", binary.M2FromChirpMassQ(mc, q)/binary.MSUN)
	fmt.Printf("m1 from (Mc, eta) : %.4f Msun\n", binary.M1FromChirpMassEta(mc, eta)/binary.MSUN)
	fmt.Printf("m2 from (Mc, eta) : %.4f Msun\n", binary.M2FromChirpMassEta(mc, eta)/binary.MSUN)

	a1 := 0.3
	a2 := 0.2
	tilt1 := 0.5236 // 30 deg
	tilt2 := 1.0472 // 60 deg

	fmt.Printf("chi_eff         : %.4f\n", binary.ChiEff(m1, m2, a1, a2, tilt1, tilt2))
	fmt.Printf("chi_p           : %.4f\n", binary.ChiP(m1, m2, a1, a2, tilt1, tilt2))
}
