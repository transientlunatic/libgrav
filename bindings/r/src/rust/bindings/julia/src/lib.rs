//! C-compatible shared library for the Julia `ccall` interface.
//!
//! All functions:
//!   - are prefixed with `puddin_` to avoid symbol clashes
//!   - accept and return `f64` values in **SI units** (kg for mass, radians
//!     for angles, dimensionless otherwise)
//!
//! Julia callers use `ccall` to invoke these functions directly.  Because
//! Julia has excellent broadcasting, the C API is **scalar only** — array
//! operations are handled by the Julia layer with `f.(args...)` syntax.
//!
//! Build:
//! ```sh
//! cargo build --release -p puddin-julia
//! # → target/release/libpuddin_julia.{so,dylib,dll}
//! ```

use puddin::binary;
use uom::si::f64::Mass;
use uom::si::mass::kilogram;

// ── helpers ───────────────────────────────────────────────────────────────────

#[inline]
fn kg(v: f64) -> Mass {
    Mass::new::<kilogram>(v)
}

// ── binary parameters ─────────────────────────────────────────────────────────

/// Total mass $M = m_1 + m_2$ (kg).
#[no_mangle]
pub extern "C" fn puddin_total_mass(m1_kg: f64, m2_kg: f64) -> f64 {
    binary::total_mass(kg(m1_kg), kg(m2_kg)).get::<kilogram>()
}

/// Mass ratio $q = m_2 / m_1$ (dimensionless).  Requires $m_1 \geq m_2$.
#[no_mangle]
pub extern "C" fn puddin_mass_ratio(m1_kg: f64, m2_kg: f64) -> f64 {
    binary::mass_ratio(kg(m1_kg), kg(m2_kg))
}

/// Symmetric mass ratio $\eta = m_1 m_2 / M^2$ (dimensionless).
#[no_mangle]
pub extern "C" fn puddin_symmetric_mass_ratio(m1_kg: f64, m2_kg: f64) -> f64 {
    binary::symmetric_mass_ratio(kg(m1_kg), kg(m2_kg))
}

/// Chirp mass $\mathcal{M} = (m_1 m_2)^{3/5} / M^{1/5}$ (kg).
#[no_mangle]
pub extern "C" fn puddin_chirp_mass(m1_kg: f64, m2_kg: f64) -> f64 {
    binary::chirp_mass(kg(m1_kg), kg(m2_kg)).get::<kilogram>()
}

/// Primary mass $m_1$ (kg) recovered from chirp mass and mass ratio $q = m_2/m_1$.
///
/// Requires $q \in (0, 1]$.
#[no_mangle]
pub extern "C" fn puddin_m1_from_mc_q(mc_kg: f64, q: f64) -> f64 {
    binary::masses_from_chirp_mass_q(kg(mc_kg), q)
        .0
        .get::<kilogram>()
}

/// Secondary mass $m_2$ (kg) recovered from chirp mass and mass ratio $q = m_2/m_1$.
///
/// Requires $q \in (0, 1]$.
#[no_mangle]
pub extern "C" fn puddin_m2_from_mc_q(mc_kg: f64, q: f64) -> f64 {
    binary::masses_from_chirp_mass_q(kg(mc_kg), q)
        .1
        .get::<kilogram>()
}

/// Primary mass $m_1$ (kg) recovered from chirp mass and symmetric mass ratio $\eta$.
///
/// Requires $\eta \in (0, 1/4]$.
#[no_mangle]
pub extern "C" fn puddin_m1_from_mc_eta(mc_kg: f64, eta: f64) -> f64 {
    binary::masses_from_chirp_mass_eta(kg(mc_kg), eta)
        .0
        .get::<kilogram>()
}

/// Secondary mass $m_2$ (kg) recovered from chirp mass and symmetric mass ratio $\eta$.
///
/// Requires $\eta \in (0, 1/4]$.
#[no_mangle]
pub extern "C" fn puddin_m2_from_mc_eta(mc_kg: f64, eta: f64) -> f64 {
    binary::masses_from_chirp_mass_eta(kg(mc_kg), eta)
        .1
        .get::<kilogram>()
}

/// Effective inspiral spin $\chi_\mathrm{eff} \in [-1, 1]$.
///
/// # Arguments
/// - `m1_kg`, `m2_kg` — component masses in kg
/// - `a1`, `a2`       — dimensionless spin magnitudes (0–1)
/// - `tilt1`, `tilt2` — spin tilt angles in radians (0–π)
#[no_mangle]
pub extern "C" fn puddin_chi_eff(
    m1_kg: f64,
    m2_kg: f64,
    a1: f64,
    a2: f64,
    tilt1: f64,
    tilt2: f64,
) -> f64 {
    binary::chi_eff(kg(m1_kg), kg(m2_kg), a1, a2, tilt1, tilt2)
}

/// Effective precession spin $\chi_p \in [0, 1]$.
///
/// **Requires** $m_1 \geq m_2$.
///
/// # Arguments
/// - `m1_kg`, `m2_kg` — component masses in kg ($m_1 \geq m_2$)
/// - `a1`, `a2`       — dimensionless spin magnitudes (0–1)
/// - `tilt1`, `tilt2` — spin tilt angles in radians (0–π)
#[no_mangle]
pub extern "C" fn puddin_chi_p(
    m1_kg: f64,
    m2_kg: f64,
    a1: f64,
    a2: f64,
    tilt1: f64,
    tilt2: f64,
) -> f64 {
    binary::chi_p(kg(m1_kg), kg(m2_kg), a1, a2, tilt1, tilt2)
}
