//! WebAssembly bindings for Grav.
//!
//! This module exposes the core Grav functions to JavaScript and TypeScript
//! via `wasm-bindgen`.  All functions accept and return `Float64Array`s in SI
//! units (kg for mass, radians for angles, dimensionless otherwise).
//!
//! Build with:
//! ```sh
//! wasm-pack build --target bundler bindings/wasm
//! ```
//!
//! The output package in `bindings/wasm/pkg/` includes TypeScript declaration
//! files (`.d.ts`) generated automatically by `wasm-bindgen`.

use grav::binary;
use uom::si::f64::Mass;
use uom::si::mass::kilogram;
use wasm_bindgen::prelude::*;

// ── helpers ───────────────────────────────────────────────────────────────────

fn to_masses(arr: &[f64]) -> Vec<Mass> {
    arr.iter().map(|&v| Mass::new::<kilogram>(v)).collect()
}

// ── binary parameters ─────────────────────────────────────────────────────────

/// Total mass $M = m_1 + m_2$ (kg).
///
/// @param m1 - Component mass 1 in kilograms.
/// @param m2 - Component mass 2 in kilograms.
/// @returns Total mass in kilograms.
#[wasm_bindgen]
pub fn total_mass(m1: Vec<f64>, m2: Vec<f64>) -> Vec<f64> {
    to_masses(&m1)
        .into_iter()
        .zip(to_masses(&m2))
        .map(|(a, b)| binary::total_mass(a, b).get::<kilogram>())
        .collect()
}

/// Mass ratio $q = m_2 / m_1$ (dimensionless).
///
/// @param m1 - Primary mass in kilograms ($m_1 \geq m_2$).
/// @param m2 - Secondary mass in kilograms.
/// @returns Dimensionless mass ratio.
#[wasm_bindgen]
pub fn mass_ratio(m1: Vec<f64>, m2: Vec<f64>) -> Vec<f64> {
    to_masses(&m1)
        .into_iter()
        .zip(to_masses(&m2))
        .map(|(a, b)| binary::mass_ratio(a, b))
        .collect()
}

/// Symmetric mass ratio $\eta = m_1 m_2 / M^2 \in (0, 1/4]$ (dimensionless).
///
/// @param m1 - Component mass 1 in kilograms.
/// @param m2 - Component mass 2 in kilograms.
/// @returns Dimensionless symmetric mass ratio.
#[wasm_bindgen]
pub fn symmetric_mass_ratio(m1: Vec<f64>, m2: Vec<f64>) -> Vec<f64> {
    to_masses(&m1)
        .into_iter()
        .zip(to_masses(&m2))
        .map(|(a, b)| binary::symmetric_mass_ratio(a, b))
        .collect()
}

/// Chirp mass $\mathcal{M} = (m_1 m_2)^{3/5} / M^{1/5}$ (kg).
///
/// @param m1 - Component mass 1 in kilograms.
/// @param m2 - Component mass 2 in kilograms.
/// @returns Chirp mass in kilograms.
#[wasm_bindgen]
pub fn chirp_mass(m1: Vec<f64>, m2: Vec<f64>) -> Vec<f64> {
    to_masses(&m1)
        .into_iter()
        .zip(to_masses(&m2))
        .map(|(a, b)| binary::chirp_mass(a, b).get::<kilogram>())
        .collect()
}

/// Component masses $(m_1, m_2)$ from chirp mass $\mathcal{M}$ and mass ratio $q$.
///
/// @param mc - Chirp mass in kilograms.
/// @param q  - Mass ratio q = m₂/m₁ ∈ (0, 1].
/// @returns Component mass m₁ in kilograms (same length as input).
#[wasm_bindgen]
pub fn masses_from_chirp_mass_q_m1(mc: Vec<f64>, q: Vec<f64>) -> Vec<f64> {
    to_masses(&mc)
        .into_iter()
        .zip(q.iter())
        .map(|(mc_val, &q_val)| {
            binary::masses_from_chirp_mass_q(mc_val, q_val)
                .0
                .get::<kilogram>()
        })
        .collect()
}

/// Component masses $(m_1, m_2)$ from chirp mass $\mathcal{M}$ and mass ratio $q$.
///
/// @param mc - Chirp mass in kilograms.
/// @param q  - Mass ratio q = m₂/m₁ ∈ (0, 1].
/// @returns Component mass m₂ in kilograms (same length as input).
#[wasm_bindgen]
pub fn masses_from_chirp_mass_q_m2(mc: Vec<f64>, q: Vec<f64>) -> Vec<f64> {
    to_masses(&mc)
        .into_iter()
        .zip(q.iter())
        .map(|(mc_val, &q_val)| {
            binary::masses_from_chirp_mass_q(mc_val, q_val)
                .1
                .get::<kilogram>()
        })
        .collect()
}

/// Component masses $(m_1, m_2)$ from chirp mass $\mathcal{M}$ and symmetric mass ratio $\eta$.
///
/// @param mc  - Chirp mass in kilograms.
/// @param eta - Symmetric mass ratio η = m₁m₂/M² ∈ (0, 0.25].
/// @returns Component mass m₁ in kilograms (same length as input).
#[wasm_bindgen]
pub fn masses_from_chirp_mass_eta_m1(mc: Vec<f64>, eta: Vec<f64>) -> Vec<f64> {
    to_masses(&mc)
        .into_iter()
        .zip(eta.iter())
        .map(|(mc_val, &eta_val)| {
            binary::masses_from_chirp_mass_eta(mc_val, eta_val)
                .0
                .get::<kilogram>()
        })
        .collect()
}

/// Component masses $(m_1, m_2)$ from chirp mass $\mathcal{M}$ and symmetric mass ratio $\eta$.
///
/// @param mc  - Chirp mass in kilograms.
/// @param eta - Symmetric mass ratio η = m₁m₂/M² ∈ (0, 0.25].
/// @returns Component mass m₂ in kilograms (same length as input).
#[wasm_bindgen]
pub fn masses_from_chirp_mass_eta_m2(mc: Vec<f64>, eta: Vec<f64>) -> Vec<f64> {
    to_masses(&mc)
        .into_iter()
        .zip(eta.iter())
        .map(|(mc_val, &eta_val)| {
            binary::masses_from_chirp_mass_eta(mc_val, eta_val)
                .1
                .get::<kilogram>()
        })
        .collect()
}

/// Effective inspiral spin $\chi_\mathrm{eff} \in [-1, 1]$.
///
/// @param m1    - Component mass 1 in kilograms.
/// @param m2    - Component mass 2 in kilograms.
/// @param a1    - Dimensionless spin magnitude of body 1 (0–1).
/// @param a2    - Dimensionless spin magnitude of body 2 (0–1).
/// @param tilt1 - Spin tilt angle of body 1 (radians, 0–π).
/// @param tilt2 - Spin tilt angle of body 2 (radians, 0–π).
/// @returns Dimensionless effective inspiral spin.
#[wasm_bindgen]
pub fn chi_eff(
    m1: Vec<f64>,
    m2: Vec<f64>,
    a1: Vec<f64>,
    a2: Vec<f64>,
    tilt1: Vec<f64>,
    tilt2: Vec<f64>,
) -> Vec<f64> {
    let n = m1.len();
    let masses1 = to_masses(&m1);
    let masses2 = to_masses(&m2);
    (0..n)
        .map(|i| binary::chi_eff(masses1[i], masses2[i], a1[i], a2[i], tilt1[i], tilt2[i]))
        .collect()
}

/// Effective precession spin $\chi_p \in [0, 1]$.
///
/// **Note:** requires $m_1 \geq m_2$.
///
/// @param m1    - Primary mass in kilograms ($m_1 \geq m_2$).
/// @param m2    - Secondary mass in kilograms.
/// @param a1    - Dimensionless spin magnitude of body 1 (0–1).
/// @param a2    - Dimensionless spin magnitude of body 2 (0–1).
/// @param tilt1 - Spin tilt angle of body 1 (radians, 0–π).
/// @param tilt2 - Spin tilt angle of body 2 (radians, 0–π).
/// @returns Dimensionless effective precession spin.
#[wasm_bindgen]
pub fn chi_p(
    m1: Vec<f64>,
    m2: Vec<f64>,
    a1: Vec<f64>,
    a2: Vec<f64>,
    tilt1: Vec<f64>,
    tilt2: Vec<f64>,
) -> Vec<f64> {
    let n = m1.len();
    let masses1 = to_masses(&m1);
    let masses2 = to_masses(&m2);
    (0..n)
        .map(|i| binary::chi_p(masses1[i], masses2[i], a1[i], a2[i], tilt1[i], tilt2[i]))
        .collect()
}

// ── spin components ──────────────────────────────────────────────────────────
//
// One function per output component, matching the split-return convention
// used above for `masses_from_chirp_mass_q_m1` / `_m2`.

macro_rules! spin_components_component {
    ($name:ident, $doc:literal, $index:tt) => {
        #[doc = $doc]
        ///
        /// @param a1    - Dimensionless spin magnitude of body 1 (0–1).
        /// @param a2    - Dimensionless spin magnitude of body 2 (0–1).
        /// @param tilt1 - Spin tilt angle of body 1 (radians, 0–π).
        /// @param tilt2 - Spin tilt angle of body 2 (radians, 0–π).
        /// @param phi12 - Azimuthal angle of spin 2 relative to spin 1 (radians).
        #[wasm_bindgen]
        pub fn $name(
            a1: Vec<f64>,
            a2: Vec<f64>,
            tilt1: Vec<f64>,
            tilt2: Vec<f64>,
            phi12: Vec<f64>,
        ) -> Vec<f64> {
            let n = a1.len();
            (0..n)
                .map(|i| binary::spin_components(a1[i], a2[i], tilt1[i], tilt2[i], phi12[i]).$index)
                .collect()
        }
    };
}

spin_components_component!(spin_components_s1x, "Cartesian spin component S1x (L-frame, dimensionless).", 0);
spin_components_component!(spin_components_s1y, "Cartesian spin component S1y (L-frame, dimensionless).", 1);
spin_components_component!(spin_components_s1z, "Cartesian spin component S1z (L-frame, dimensionless).", 2);
spin_components_component!(spin_components_s2x, "Cartesian spin component S2x (L-frame, dimensionless).", 3);
spin_components_component!(spin_components_s2y, "Cartesian spin component S2y (L-frame, dimensionless).", 4);
spin_components_component!(spin_components_s2z, "Cartesian spin component S2z (L-frame, dimensionless).", 5);

// ── orbital angular momentum ─────────────────────────────────────────────────

/// Newtonian orbital angular momentum magnitude $|L_N|$ (kg m² s⁻¹).
///
/// @param m1    - Component mass 1 in kilograms.
/// @param m2    - Component mass 2 in kilograms.
/// @param f_ref - Reference GW frequency in Hz.
/// @returns |L_N| in kg m² s⁻¹.
#[wasm_bindgen]
pub fn orbital_angular_momentum(m1: Vec<f64>, m2: Vec<f64>, f_ref: Vec<f64>) -> Vec<f64> {
    let masses1 = to_masses(&m1);
    let masses2 = to_masses(&m2);
    masses1
        .into_iter()
        .zip(masses2)
        .zip(f_ref.iter())
        .map(|((a, b), &f)| binary::orbital_angular_momentum(a, b, f))
        .collect()
}

// ── precessing spin frame transform ──────────────────────────────────────────
//
// A from-scratch reimplementation of LALSimulation's
// SimInspiralTransformPrecessingNewInitialConditions (no LALSuite
// dependency); matches its output to within floating-point precision.

macro_rules! transform_precessing_spins_component {
    ($name:ident, $doc:literal, $index:tt) => {
        #[doc = $doc]
        ///
        /// @param theta_jn - Inclination of J relative to the line of sight (radians).
        /// @param phi_jl   - Azimuth of L_N about J (radians).
        /// @param tilt1    - Spin tilt angle of body 1 from L_N (radians, 0–π).
        /// @param tilt2    - Spin tilt angle of body 2 from L_N (radians, 0–π).
        /// @param phi12    - Azimuthal angle of spin 2 relative to spin 1 (radians).
        /// @param a1       - Dimensionless spin magnitude of body 1 (0–1).
        /// @param a2       - Dimensionless spin magnitude of body 2 (0–1).
        /// @param m1       - Component mass 1 in kilograms.
        /// @param m2       - Component mass 2 in kilograms.
        /// @param f_ref    - Reference GW frequency in Hz (nonzero).
        /// @param phase    - Reference orbital phase (radians).
        #[wasm_bindgen]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            theta_jn: Vec<f64>,
            phi_jl: Vec<f64>,
            tilt1: Vec<f64>,
            tilt2: Vec<f64>,
            phi12: Vec<f64>,
            a1: Vec<f64>,
            a2: Vec<f64>,
            m1: Vec<f64>,
            m2: Vec<f64>,
            f_ref: Vec<f64>,
            phase: Vec<f64>,
        ) -> Vec<f64> {
            let n = theta_jn.len();
            let masses1 = to_masses(&m1);
            let masses2 = to_masses(&m2);
            (0..n)
                .map(|i| {
                    binary::transform_precessing_spins(
                        theta_jn[i],
                        phi_jl[i],
                        tilt1[i],
                        tilt2[i],
                        phi12[i],
                        a1[i],
                        a2[i],
                        masses1[i],
                        masses2[i],
                        f_ref[i],
                        phase[i],
                    )
                    .$index
                })
                .collect()
        }
    };
}

transform_precessing_spins_component!(transform_precessing_spins_iota, "Inclination of L_N relative to the line of sight, after the J→L rotation (radians).", 0);
transform_precessing_spins_component!(transform_precessing_spins_s1x, "Cartesian spin component S1x after the precessing-spin frame transform.", 1);
transform_precessing_spins_component!(transform_precessing_spins_s1y, "Cartesian spin component S1y after the precessing-spin frame transform.", 2);
transform_precessing_spins_component!(transform_precessing_spins_s1z, "Cartesian spin component S1z after the precessing-spin frame transform.", 3);
transform_precessing_spins_component!(transform_precessing_spins_s2x, "Cartesian spin component S2x after the precessing-spin frame transform.", 4);
transform_precessing_spins_component!(transform_precessing_spins_s2y, "Cartesian spin component S2y after the precessing-spin frame transform.", 5);
transform_precessing_spins_component!(transform_precessing_spins_s2z, "Cartesian spin component S2z after the precessing-spin frame transform.", 6);
