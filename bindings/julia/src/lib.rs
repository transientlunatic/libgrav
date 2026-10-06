//! C-compatible shared library for the Julia `ccall` interface.
//!
//! All functions:
//!   - are prefixed with `grav_` to avoid symbol clashes
//!   - accept and return `f64` values in **SI units** (kg for mass, radians
//!     for angles, dimensionless otherwise)
//!
//! Julia callers use `ccall` to invoke these functions directly.  Because
//! Julia has excellent broadcasting, the C API is **scalar only** — array
//! operations are handled by the Julia layer with `f.(args...)` syntax.
//!
//! Build:
//! ```sh
//! cargo build --release -p grav-capi
//! # → target/release/libgrav.{so,dylib,dll}
//! ```

use grav::binary;
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
pub extern "C" fn grav_total_mass(m1_kg: f64, m2_kg: f64) -> f64 {
    binary::total_mass(kg(m1_kg), kg(m2_kg)).get::<kilogram>()
}

/// Mass ratio $q = m_2 / m_1$ (dimensionless).  Requires $m_1 \geq m_2$.
#[no_mangle]
pub extern "C" fn grav_mass_ratio(m1_kg: f64, m2_kg: f64) -> f64 {
    binary::mass_ratio(kg(m1_kg), kg(m2_kg))
}

/// Symmetric mass ratio $\eta = m_1 m_2 / M^2$ (dimensionless).
#[no_mangle]
pub extern "C" fn grav_symmetric_mass_ratio(m1_kg: f64, m2_kg: f64) -> f64 {
    binary::symmetric_mass_ratio(kg(m1_kg), kg(m2_kg))
}

/// Chirp mass $\mathcal{M} = (m_1 m_2)^{3/5} / M^{1/5}$ (kg).
#[no_mangle]
pub extern "C" fn grav_chirp_mass(m1_kg: f64, m2_kg: f64) -> f64 {
    binary::chirp_mass(kg(m1_kg), kg(m2_kg)).get::<kilogram>()
}

/// Primary mass $m_1$ (kg) recovered from chirp mass and mass ratio $q = m_2/m_1$.
///
/// Requires $q \in (0, 1]$.
#[no_mangle]
pub extern "C" fn grav_m1_from_mc_q(mc_kg: f64, q: f64) -> f64 {
    binary::masses_from_chirp_mass_q(kg(mc_kg), q)
        .0
        .get::<kilogram>()
}

/// Secondary mass $m_2$ (kg) recovered from chirp mass and mass ratio $q = m_2/m_1$.
///
/// Requires $q \in (0, 1]$.
#[no_mangle]
pub extern "C" fn grav_m2_from_mc_q(mc_kg: f64, q: f64) -> f64 {
    binary::masses_from_chirp_mass_q(kg(mc_kg), q)
        .1
        .get::<kilogram>()
}

/// Primary mass $m_1$ (kg) recovered from chirp mass and symmetric mass ratio $\eta$.
///
/// Requires $\eta \in (0, 1/4]$.
#[no_mangle]
pub extern "C" fn grav_m1_from_mc_eta(mc_kg: f64, eta: f64) -> f64 {
    binary::masses_from_chirp_mass_eta(kg(mc_kg), eta)
        .0
        .get::<kilogram>()
}

/// Secondary mass $m_2$ (kg) recovered from chirp mass and symmetric mass ratio $\eta$.
///
/// Requires $\eta \in (0, 1/4]$.
#[no_mangle]
pub extern "C" fn grav_m2_from_mc_eta(mc_kg: f64, eta: f64) -> f64 {
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
pub extern "C" fn grav_chi_eff(
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
pub extern "C" fn grav_chi_p(
    m1_kg: f64,
    m2_kg: f64,
    a1: f64,
    a2: f64,
    tilt1: f64,
    tilt2: f64,
) -> f64 {
    binary::chi_p(kg(m1_kg), kg(m2_kg), a1, a2, tilt1, tilt2)
}

// ── spin components ──────────────────────────────────────────────────────────
//
// One function per output component, matching the split-return convention
// used above for `grav_m1_from_mc_q` / `grav_m2_from_mc_q`.

macro_rules! spin_components_component {
    ($name:ident, $index:tt) => {
        /// Cartesian spin component in the L-frame; see `grav_spin_components_s1x`
        /// for the full parameter description.
        #[no_mangle]
        pub extern "C" fn $name(a1: f64, a2: f64, tilt1: f64, tilt2: f64, phi12: f64) -> f64 {
            binary::spin_components(a1, a2, tilt1, tilt2, phi12).$index
        }
    };
}

spin_components_component!(grav_spin_components_s1x, 0);
spin_components_component!(grav_spin_components_s1y, 1);
spin_components_component!(grav_spin_components_s1z, 2);
spin_components_component!(grav_spin_components_s2x, 3);
spin_components_component!(grav_spin_components_s2y, 4);
spin_components_component!(grav_spin_components_s2z, 5);

// ── orbital angular momentum ─────────────────────────────────────────────────

/// Newtonian orbital angular momentum magnitude $|L_N|$ (kg m² s⁻¹).
#[no_mangle]
pub extern "C" fn grav_orbital_angular_momentum(m1_kg: f64, m2_kg: f64, f_ref: f64) -> f64 {
    binary::orbital_angular_momentum(kg(m1_kg), kg(m2_kg), f_ref)
}

// ── precessing spin frame transform ──────────────────────────────────────────

macro_rules! transform_precessing_spins_component {
    ($name:ident, $index:tt) => {
        /// Component of the precessing-spin frame transform; see
        /// `grav_transform_precessing_spins_iota` for the full parameter
        /// description.
        #[no_mangle]
        #[allow(clippy::too_many_arguments)]
        pub extern "C" fn $name(
            theta_jn: f64,
            phi_jl: f64,
            tilt1: f64,
            tilt2: f64,
            phi12: f64,
            a1: f64,
            a2: f64,
            m1_kg: f64,
            m2_kg: f64,
            f_ref: f64,
            phase: f64,
        ) -> f64 {
            binary::transform_precessing_spins(
                theta_jn,
                phi_jl,
                tilt1,
                tilt2,
                phi12,
                a1,
                a2,
                kg(m1_kg),
                kg(m2_kg),
                f_ref,
                phase,
            )
            .$index
        }
    };
}

// Component of L_N's inclination / the precessing-spin Cartesian transform.
//
// Arguments (all functions in this group):
// - `theta_jn` — inclination of J relative to the line of sight (radians)
// - `phi_jl`   — azimuth of L_N about J (radians)
// - `tilt1`, `tilt2` — spin tilt angles from L_N (radians)
// - `phi12`    — azimuthal angle of spin 2 relative to spin 1 (radians)
// - `a1`, `a2` — dimensionless spin magnitudes (0–1)
// - `m1_kg`, `m2_kg` — component masses (kg)
// - `f_ref`    — reference GW frequency (Hz), must be nonzero
// - `phase`    — reference orbital phase (radians)
//
// `_iota` returns the inclination of L_N relative to the line of sight
// (radians); `_s1x`.._s2z return the Cartesian spin components.
transform_precessing_spins_component!(grav_transform_precessing_spins_iota, 0);
transform_precessing_spins_component!(grav_transform_precessing_spins_s1x, 1);
transform_precessing_spins_component!(grav_transform_precessing_spins_s1y, 2);
transform_precessing_spins_component!(grav_transform_precessing_spins_s1z, 3);
transform_precessing_spins_component!(grav_transform_precessing_spins_s2x, 4);
transform_precessing_spins_component!(grav_transform_precessing_spins_s2y, 5);
transform_precessing_spins_component!(grav_transform_precessing_spins_s2z, 6);
