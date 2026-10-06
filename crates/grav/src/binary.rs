//! Binary system parameters.
//!
//! This module provides functions for converting between the parameterisations
//! commonly used to describe compact binary systems in gravitational-wave
//! astronomy.  All quantities are expressed using SI units via the [`uom`]
//! crate, which enforces dimensional correctness at compile time.
//!
//! # Conventions
//!
//! - Component masses are labelled $m_1 \geq m_2 > 0$.
//! - Spin magnitudes are $a_i \in [0, 1]$ (dimensionless, normalised to the
//!   Kerr maximum).
//! - Spin tilt angles $\theta_i$ are measured from the orbital angular momentum
//!   axis, $\theta_i \in [0, \pi]$.
//! - Angles are in radians.

use uom::si::f64::*;
use uom::si::mass::kilogram;

// ── Total mass ───────────────────────────────────────────────────────────────

/// Total mass $M = m_1 + m_2$.
///
/// # Examples
///
/// ```
/// use uom::si::f64::Mass;
/// use uom::si::mass::kilogram;
/// use grav::binary::total_mass;
///
/// const MSUN: f64 = 1.988_416e30;
/// let m1 = Mass::new::<kilogram>(30.0 * MSUN);
/// let m2 = Mass::new::<kilogram>(20.0 * MSUN);
/// let m = total_mass(m1, m2);
/// assert!((m.get::<kilogram>() - 50.0 * MSUN).abs() < 1e6);
/// ```
pub fn total_mass(m1: Mass, m2: Mass) -> Mass {
    m1 + m2
}

// ── Mass ratio ───────────────────────────────────────────────────────────────

/// Mass ratio $q = m_2 / m_1$, where $m_1 \geq m_2$ so $q \in (0, 1]$.
///
/// # Examples
///
/// ```
/// use uom::si::f64::Mass;
/// use uom::si::mass::kilogram;
/// use grav::binary::mass_ratio;
///
/// const MSUN: f64 = 1.988_416e30;
/// let m1 = Mass::new::<kilogram>(30.0 * MSUN);
/// let m2 = Mass::new::<kilogram>(15.0 * MSUN);
/// let q = mass_ratio(m1, m2);
/// assert!((q - 0.5).abs() < 1e-10);
/// ```
pub fn mass_ratio(m1: Mass, m2: Mass) -> f64 {
    m2.get::<kilogram>() / m1.get::<kilogram>()
}

// ── Symmetric mass ratio ─────────────────────────────────────────────────────

/// Symmetric mass ratio $\eta = m_1 m_2 / M^2 \in (0, 1/4]$.
///
/// Equal-mass systems have $\eta = 1/4$; highly asymmetric systems have
/// $\eta \to 0$.
///
/// # Examples
///
/// ```
/// use uom::si::f64::Mass;
/// use uom::si::mass::kilogram;
/// use grav::binary::symmetric_mass_ratio;
///
/// const MSUN: f64 = 1.988_416e30;
/// let m1 = Mass::new::<kilogram>(30.0 * MSUN);
/// let m2 = Mass::new::<kilogram>(30.0 * MSUN);
/// let eta = symmetric_mass_ratio(m1, m2);
/// assert!((eta - 0.25).abs() < 1e-10);
/// ```
pub fn symmetric_mass_ratio(m1: Mass, m2: Mass) -> f64 {
    let m1_kg = m1.get::<kilogram>();
    let m2_kg = m2.get::<kilogram>();
    let m_kg = m1_kg + m2_kg;
    (m1_kg * m2_kg) / (m_kg * m_kg)
}

// ── Chirp mass ───────────────────────────────────────────────────────────────

/// Chirp mass $\mathcal{M} = (m_1 m_2)^{3/5} / M^{1/5}$.
///
/// The chirp mass is the combination of masses that governs the leading-order
/// gravitational-wave frequency evolution during inspiral.  It is always less
/// than or equal to the total mass.
///
/// # Examples
///
/// ```
/// use uom::si::f64::Mass;
/// use uom::si::mass::kilogram;
/// use grav::binary::chirp_mass;
///
/// const MSUN: f64 = 1.988_416e30;
/// let m1 = Mass::new::<kilogram>(30.0 * MSUN);
/// let m2 = Mass::new::<kilogram>(30.0 * MSUN);
/// let mc = chirp_mass(m1, m2);
/// // For equal masses: Mc = M * (1/4)^(3/5) = 2m * (1/4)^(3/5)
/// let expected_kg = 2.0 * 30.0 * MSUN * 0.25_f64.powf(3.0 / 5.0);
/// assert!((mc.get::<kilogram>() - expected_kg).abs() / expected_kg < 1e-10);
/// ```
pub fn chirp_mass(m1: Mass, m2: Mass) -> Mass {
    let m1_kg = m1.get::<kilogram>();
    let m2_kg = m2.get::<kilogram>();
    let m_kg = m1_kg + m2_kg;
    let mc_kg = (m1_kg * m2_kg).powf(3.0 / 5.0) / m_kg.powf(1.0 / 5.0);
    Mass::new::<kilogram>(mc_kg)
}

// ── Inverse mass transforms ──────────────────────────────────────────────────

/// Recover component masses $(m_1, m_2)$ from chirp mass $\mathcal{M}$ and
/// mass ratio $q = m_2/m_1$.
///
/// This is the inverse of `chirp_mass` + `mass_ratio`.  The returned masses
/// satisfy $m_1 \geq m_2$ whenever $q \in (0, 1]$.
///
/// # Arguments
///
/// * `mc` — chirp mass $\mathcal{M}$ (SI kg).
/// * `q`  — mass ratio $q = m_2/m_1 \in (0, 1]$.
///
/// # Examples
///
/// ```
/// use uom::si::f64::Mass;
/// use uom::si::mass::kilogram;
/// use grav::binary::{chirp_mass, mass_ratio, masses_from_chirp_mass_q};
///
/// const MSUN: f64 = 1.988_416e30;
/// let m1_in = Mass::new::<kilogram>(30.0 * MSUN);
/// let m2_in = Mass::new::<kilogram>(20.0 * MSUN);
/// let mc = chirp_mass(m1_in, m2_in);
/// let q  = mass_ratio(m1_in, m2_in);
/// let (m1_out, m2_out) = masses_from_chirp_mass_q(mc, q);
/// assert!((m1_out.get::<kilogram>() - m1_in.get::<kilogram>()).abs() / m1_in.get::<kilogram>() < 1e-10);
/// assert!((m2_out.get::<kilogram>() - m2_in.get::<kilogram>()).abs() / m2_in.get::<kilogram>() < 1e-10);
/// ```
pub fn masses_from_chirp_mass_q(mc: Mass, q: f64) -> (Mass, Mass) {
    debug_assert!(
        q > 0.0 && q <= 1.0,
        "mass ratio q must be in (0, 1], got q={q}"
    );
    let eta = q / (1.0 + q).powi(2);
    let m_kg = mc.get::<kilogram>() / eta.powf(3.0 / 5.0);
    let m1_kg = m_kg / (1.0 + q);
    let m2_kg = m1_kg * q;
    (Mass::new::<kilogram>(m1_kg), Mass::new::<kilogram>(m2_kg))
}

/// Recover component masses $(m_1, m_2)$ from chirp mass $\mathcal{M}$ and
/// symmetric mass ratio $\eta$.
///
/// This is the inverse of `chirp_mass` + `symmetric_mass_ratio`.  The
/// quadratic $x^2 - x + \eta = 0$ gives $m_1/M$ and $m_2/M$; the larger
/// root is assigned to $m_1$ so that $m_1 \geq m_2$.
///
/// For equal-mass systems ($\eta = 1/4$) the two roots coincide and
/// $m_1 = m_2$.
///
/// # Arguments
///
/// * `mc`  — chirp mass $\mathcal{M}$ (SI kg).
/// * `eta` — symmetric mass ratio $\eta \in (0, 1/4]$.
///
/// # Examples
///
/// ```
/// use uom::si::f64::Mass;
/// use uom::si::mass::kilogram;
/// use grav::binary::{chirp_mass, symmetric_mass_ratio, masses_from_chirp_mass_eta};
///
/// const MSUN: f64 = 1.988_416e30;
/// let m1_in = Mass::new::<kilogram>(30.0 * MSUN);
/// let m2_in = Mass::new::<kilogram>(20.0 * MSUN);
/// let mc  = chirp_mass(m1_in, m2_in);
/// let eta = symmetric_mass_ratio(m1_in, m2_in);
/// let (m1_out, m2_out) = masses_from_chirp_mass_eta(mc, eta);
/// assert!((m1_out.get::<kilogram>() - m1_in.get::<kilogram>()).abs() / m1_in.get::<kilogram>() < 1e-10);
/// assert!((m2_out.get::<kilogram>() - m2_in.get::<kilogram>()).abs() / m2_in.get::<kilogram>() < 1e-10);
/// ```
pub fn masses_from_chirp_mass_eta(mc: Mass, eta: f64) -> (Mass, Mass) {
    debug_assert!(
        eta > 0.0 && eta <= 0.25 + 1e-12,
        "symmetric mass ratio eta must be in (0, 0.25], got eta={eta}"
    );
    let eta = eta.min(0.25); // clamp floating-point noise at equal mass
    let m_kg = mc.get::<kilogram>() / eta.powf(3.0 / 5.0);
    let disc = (1.0 - 4.0 * eta).max(0.0).sqrt();
    let m1_kg = m_kg * (1.0 + disc) / 2.0;
    let m2_kg = m_kg * (1.0 - disc) / 2.0;
    (Mass::new::<kilogram>(m1_kg), Mass::new::<kilogram>(m2_kg))
}

// ── Effective inspiral spin ──────────────────────────────────────────────────

/// Effective inspiral spin parameter
/// $\chi_\mathrm{eff} = (m_1 a_1 \cos\theta_1 + m_2 a_2 \cos\theta_2) / M$.
///
/// $\chi_\mathrm{eff} \in [-1, 1]$ and is approximately conserved through
/// inspiral at 1.5 post-Newtonian order.
///
/// # Arguments
///
/// * `m1`, `m2` — component masses ($m_1 \geq m_2$).
/// * `a1`, `a2` — dimensionless spin magnitudes $\in [0, 1]$.
/// * `tilt1`, `tilt2` — spin tilt angles (radians) with respect to the orbital
///   angular momentum axis.
///
/// # Examples
///
/// ```
/// use uom::si::f64::Mass;
/// use uom::si::mass::kilogram;
/// use grav::binary::chi_eff;
/// use std::f64::consts::PI;
///
/// const MSUN: f64 = 1.988_416e30;
/// let m1 = Mass::new::<kilogram>(30.0 * MSUN);
/// let m2 = Mass::new::<kilogram>(30.0 * MSUN);
/// // Both spins aligned, magnitude 0.5 -> chi_eff = 0.5
/// let x = chi_eff(m1, m2, 0.5, 0.5, 0.0, 0.0);
/// assert!((x - 0.5).abs() < 1e-10);
///
/// // Anti-aligned -> chi_eff = -0.5
/// let x = chi_eff(m1, m2, 0.5, 0.5, PI, PI);
/// assert!((x - (-0.5)).abs() < 1e-10);
/// ```
pub fn chi_eff(m1: Mass, m2: Mass, a1: f64, a2: f64, tilt1: f64, tilt2: f64) -> f64 {
    let m1_kg = m1.get::<kilogram>();
    let m2_kg = m2.get::<kilogram>();
    let m_kg = m1_kg + m2_kg;
    (m1_kg * a1 * tilt1.cos() + m2_kg * a2 * tilt2.cos()) / m_kg
}

// ── Effective precession spin ────────────────────────────────────────────────

/// Effective precession spin parameter
/// $\chi_p = \max\!\bigl(a_1 \sin\theta_1,\; \tfrac{3+4q}{4(1+q)} q \, a_2 \sin\theta_2\bigr)$
///
/// as defined in [Hannam et al. (2014)](https://doi.org/10.1103/PhysRevLett.113.151101).
/// $\chi_p \in [0, 1]$.
///
/// # Arguments
///
/// * `m1`, `m2` — component masses.  **Must satisfy $m_1 \geq m_2$**; the
///   formula is undefined (and not bounded) for $m_2 > m_1$.
/// * `a1`, `a2` — dimensionless spin magnitudes $\in [0, 1]$.
/// * `tilt1`, `tilt2` — spin tilt angles (radians).
///
/// # Examples
///
/// ```
/// use uom::si::f64::Mass;
/// use uom::si::mass::kilogram;
/// use grav::binary::chi_p;
/// use std::f64::consts::FRAC_PI_2;
///
/// const MSUN: f64 = 1.988_416e30;
/// let m1 = Mass::new::<kilogram>(30.0 * MSUN);
/// let m2 = Mass::new::<kilogram>(30.0 * MSUN);
/// // Primary spin fully in-plane, secondary aligned -> chi_p = a1 sin(pi/2) = 1.0
/// let x = chi_p(m1, m2, 1.0, 0.0, FRAC_PI_2, 0.0);
/// assert!((x - 1.0).abs() < 1e-10);
/// ```
pub fn chi_p(m1: Mass, m2: Mass, a1: f64, a2: f64, tilt1: f64, tilt2: f64) -> f64 {
    debug_assert!(
        m1.get::<kilogram>() >= m2.get::<kilogram>(),
        "chi_p requires m1 >= m2 (got m1={}, m2={})",
        m1.get::<kilogram>(),
        m2.get::<kilogram>()
    );
    let q = mass_ratio(m1, m2); // m2/m1 <= 1
    let term1 = a1 * tilt1.sin();
    let term2 = (3.0 + 4.0 * q) / (4.0 * (1.0 + q)) * q * a2 * tilt2.sin();
    term1.max(term2)
}

// ── Spin components ──────────────────────────────────────────────────────────

/// Decompose spin tilts and azimuths into Cartesian components in the L-frame.
///
/// Converts from the bilby / LALInference spin parameterisation
/// (dimensionless magnitude + tilt angle + relative azimuth) to the
/// Cartesian spin components used by LALSimulation.
///
/// The **L-frame** has its z-axis aligned with the Newtonian orbital angular
/// momentum **L̂**.  By convention spin 1 is placed in the x-z plane
/// (φ₁ = 0), so S₁ᵧ = 0 identically.  Spin 2 is rotated by φ₁₂ around the
/// z-axis relative to spin 1:
///
/// ```text
/// S₁ = a₁ (sin θ₁,  0,              cos θ₁)
/// S₂ = a₂ (sin θ₂ cos φ₁₂,  sin θ₂ sin φ₁₂,  cos θ₂)
/// ```
///
/// # Arguments
///
/// * `a1`, `a2`    — dimensionless spin magnitudes χ₁, χ₂ ∈ [0, 1].
/// * `tilt1`, `tilt2` — spin tilt angles θ₁, θ₂ ∈ [0, π] (radians).
/// * `phi12`       — azimuthal angle of spin 2 relative to spin 1 ∈ [0, 2π) (radians).
///
/// # Returns
///
/// `(S1x, S1y, S1z, S2x, S2y, S2z)` — dimensionless Cartesian components.
///
/// # Examples
///
/// ```
/// use grav::binary::spin_components;
/// use std::f64::consts::FRAC_PI_2;
///
/// // Aligned spins: both along z-axis
/// let (s1x, s1y, s1z, s2x, s2y, s2z) = spin_components(0.5, 0.3, 0.0, 0.0, 0.0);
/// assert!(s1x.abs() < 1e-14 && s1y.abs() < 1e-14);
/// assert!((s1z - 0.5).abs() < 1e-14);
/// assert!((s2z - 0.3).abs() < 1e-14);
///
/// // In-plane spin 1: tilt = π/2 → S1x = a1
/// let (s1x, s1y, s1z, _, _, _) = spin_components(0.8, 0.0, FRAC_PI_2, 0.0, 0.0);
/// assert!((s1x - 0.8).abs() < 1e-14);
/// assert!(s1z.abs() < 1e-14);
/// ```
pub fn spin_components(
    a1: f64,
    a2: f64,
    tilt1: f64,
    tilt2: f64,
    phi12: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    let s1x = a1 * tilt1.sin();
    let s1y = 0.0_f64;
    let s1z = a1 * tilt1.cos();
    let s2x = a2 * tilt2.sin() * phi12.cos();
    let s2y = a2 * tilt2.sin() * phi12.sin();
    let s2z = a2 * tilt2.cos();
    (s1x, s1y, s1z, s2x, s2y, s2z)
}

// ── Orbital angular momentum ─────────────────────────────────────────────────

/// Gravitational constant in SI units (CODATA 2014, consistent with LALSuite).
const G_SI: f64 = 6.674_30e-11; // m³ kg⁻¹ s⁻²

/// Newtonian orbital angular momentum magnitude at a reference frequency.
///
/// Computes the leading-order (Newtonian) orbital angular momentum
///
/// ```text
/// |L_N| = μ (G M)^{2/3} / (π f_ref)^{1/3}
/// ```
///
/// where μ = m₁ m₂ / M is the reduced mass and M = m₁ + m₂ is the total mass.
///
/// # Arguments
///
/// * `m1`, `m2` — component masses (SI: kg).
/// * `f_ref`    — gravitational-wave reference frequency (Hz).
///
/// # Returns
///
/// `|L_N|` in SI units (kg m² s⁻¹).
///
/// # Examples
///
/// ```
/// use uom::si::f64::Mass;
/// use uom::si::mass::kilogram;
/// use grav::binary::orbital_angular_momentum;
///
/// const MSUN: f64 = 1.988_416e30;
/// let m1 = Mass::new::<kilogram>(30.0 * MSUN);
/// let m2 = Mass::new::<kilogram>(20.0 * MSUN);
/// let l = orbital_angular_momentum(m1, m2, 20.0);
/// assert!(l > 0.0);
/// ```
pub fn orbital_angular_momentum(m1: Mass, m2: Mass, f_ref: f64) -> f64 {
    let m1_kg = m1.get::<kilogram>();
    let m2_kg = m2.get::<kilogram>();
    let m_kg = m1_kg + m2_kg;
    let mu_kg = m1_kg * m2_kg / m_kg;
    mu_kg * (G_SI * m_kg).powf(2.0 / 3.0) / (std::f64::consts::PI * f_ref).powf(1.0 / 3.0)
}

// ── Precessing spin frame transform ──────────────────────────────────────────

/// Solar mass in kilograms, matching LALSuite's `LAL_MSUN_SI` to machine
/// precision.  [`transform_precessing_spins`] must reproduce LALSimulation's
/// internal PN velocity parameter exactly, so it uses this rather than the
/// coarser `1.988_416e30` used in this module's other doctests/tests.
const LAL_MSUN_SI: f64 = 1.988_409_870_698_050_731_911_960_804_878_414_216e30;

/// Solar mass in seconds (`G M_sun / c^3`), matching LALSuite's `LAL_MTSUN_SI`.
const LAL_MTSUN_SI: f64 = 4.925_490_947_641_266_978_197_229_498_498_379_006e-6;

/// Transform bilby / LALInference precessing-spin parameters into the
/// Cartesian spin components and inclination LALSimulation waveform
/// generators expect.
///
/// This reproduces LALSimulation's
/// `XLALSimInspiralTransformPrecessingNewInitialConditions`: it constructs
/// **J** = **L_N** + **S₁** + **S₂** using the Newtonian orbital angular
/// momentum with its 1PN point-particle correction (no spin-orbit term —
/// matching upstream, see eq. 2.9 of gr-qc/9506022 and eq. 4.7 of
/// arXiv:1212.5520), then rotates from the L-frame through the J-frame into
/// the frame LALSimulation uses to generate a waveform at the given
/// reference orbital phase.
///
/// # Arguments
///
/// * `theta_jn` — inclination of **J** relative to the line of sight (rad).
/// * `phi_jl`   — azimuth of **L_N** about **J** (rad).
/// * `tilt1`, `tilt2` — spin tilt angles θ₁, θ₂ from **L_N** (rad).
/// * `phi12`    — azimuthal angle of spin 2 relative to spin 1 (rad).
/// * `a1`, `a2` — dimensionless spin magnitudes ∈ [0, 1].
/// * `m1`, `m2` — component masses (SI: kg).
/// * `f_ref`    — reference GW frequency (Hz); must be nonzero.
/// * `phase`    — reference orbital phase (rad).
///
/// # Returns
///
/// `(iota, S1x, S1y, S1z, S2x, S2y, S2z)` — inclination of **L_N** relative
/// to the line of sight, and the Cartesian spin components in the frame
/// LALSimulation waveform generators expect.
///
/// # Examples
///
/// ```
/// use uom::si::f64::Mass;
/// use uom::si::mass::kilogram;
/// use grav::binary::transform_precessing_spins;
///
/// const MSUN: f64 = 1.988_416e30;
/// let m1 = Mass::new::<kilogram>(30.0 * MSUN);
/// let m2 = Mass::new::<kilogram>(20.0 * MSUN);
///
/// // Aligned spins (tilt = 0): iota reduces to theta_jn and the in-plane
/// // components vanish, matching `spin_components`.
/// let (iota, s1x, s1y, s1z, s2x, s2y, s2z) =
///     transform_precessing_spins(0.4, 0.3, 0.0, 0.0, 1.2, 0.6, 0.4, m1, m2, 20.0, 0.0);
/// assert!((iota - 0.4).abs() < 1e-12);
/// assert!(s1x.abs() < 1e-12 && s1y.abs() < 1e-12 && (s1z - 0.6).abs() < 1e-12);
/// assert!(s2x.abs() < 1e-12 && s2y.abs() < 1e-12 && (s2z - 0.4).abs() < 1e-12);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn transform_precessing_spins(
    theta_jn: f64,
    phi_jl: f64,
    tilt1: f64,
    tilt2: f64,
    phi12: f64,
    a1: f64,
    a2: f64,
    m1: Mass,
    m2: Mass,
    f_ref: f64,
    phase: f64,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    debug_assert!(f_ref != 0.0, "f_ref must be nonzero");

    let m1_msun = m1.get::<kilogram>() / LAL_MSUN_SI;
    let m2_msun = m2.get::<kilogram>() / LAL_MSUN_SI;
    let m_total = m1_msun + m2_msun;
    let eta = symmetric_mass_ratio(m1, m2);
    let v0 = (m_total * LAL_MTSUN_SI * std::f64::consts::PI * f_ref).cbrt();

    // |L_N|, with its 1PN point-particle correction only (no spin-orbit term).
    let l_2pn = 1.5 + eta / 6.0;
    let l_mag = m_total * m_total * eta / v0 * (1.0 + v0 * v0 * l_2pn);

    // Starting frame: L_N along z; unit spin vectors relative to L_N, with
    // azimuths anchored at `phase` (S1) and `phi12 + phase` (S2).
    let (mut lnx, mut lny, mut lnz) = (0.0_f64, 0.0_f64, 1.0_f64);
    let (mut s1x, mut s1y, mut s1z) = (
        tilt1.sin() * phase.cos(),
        tilt1.sin() * phase.sin(),
        tilt1.cos(),
    );
    let (mut s2x, mut s2y, mut s2z) = (
        tilt2.sin() * (phi12 + phase).cos(),
        tilt2.sin() * (phi12 + phase).sin(),
        tilt2.cos(),
    );

    // Mass²-weighted spins to find J's direction (physical angular momentum).
    let jx = m1_msun * m1_msun * a1 * s1x + m2_msun * m2_msun * a2 * s2x;
    let jy = m1_msun * m1_msun * a1 * s1y + m2_msun * m2_msun * a2 * s2y;
    let jz = l_mag + m1_msun * m1_msun * a1 * s1z + m2_msun * m2_msun * a2 * s2z;
    let j_norm = (jx * jx + jy * jy + jz * jz).sqrt();
    let theta0 = (jz / j_norm).acos();
    let phi0 = jy.atan2(jx);

    let rotate_z = |angle: f64, x: f64, y: f64| {
        (
            x * angle.cos() - y * angle.sin(),
            x * angle.sin() + y * angle.cos(),
        )
    };
    let rotate_y = |angle: f64, x: f64, z: f64| {
        (
            x * angle.cos() + z * angle.sin(),
            -x * angle.sin() + z * angle.cos(),
        )
    };

    // Rotation 1: about z by -phi0 (LNhat, fixed along z, is unaffected).
    (s1x, s1y) = rotate_z(-phi0, s1x, s1y);
    (s2x, s2y) = rotate_z(-phi0, s2x, s2y);

    // Rotation 2: about y by -theta0, bringing Jhat onto z.
    (lnx, lnz) = rotate_y(-theta0, lnx, lnz);
    (s1x, s1z) = rotate_y(-theta0, s1x, s1z);
    (s2x, s2z) = rotate_y(-theta0, s2x, s2z);

    // Rotation 3: about z by (phi_jl - pi), placing L_N at the requested
    // azimuth about J.
    let angle3 = phi_jl - std::f64::consts::PI;
    (lnx, lny) = rotate_z(angle3, lnx, lny);
    (s1x, s1y) = rotate_z(angle3, s1x, s1y);
    (s2x, s2y) = rotate_z(angle3, s2x, s2y);

    // Observer direction N in the J-aligned frame; iota is the angle between
    // L_N (as rotated so far) and N.
    let (mut nx, mut ny, nz) = (0.0_f64, theta_jn.sin(), theta_jn.cos());
    let iota = (nx * lnx + ny * lny + nz * lnz).acos();

    // Rotations 4-5: bring L_N onto z to read off spin components there.
    let theta_lj = lnz.acos();
    let phi_l = lny.atan2(lnx);

    (s1x, s1y) = rotate_z(-phi_l, s1x, s1y);
    (s2x, s2y) = rotate_z(-phi_l, s2x, s2y);
    (nx, ny) = rotate_z(-phi_l, nx, ny);

    (s1x, s1z) = rotate_y(-theta_lj, s1x, s1z);
    (s2x, s2z) = rotate_y(-theta_lj, s2x, s2z);
    // N's z-component (line-of-sight vs L_N) isn't needed past this point.
    (nx, _) = rotate_y(-theta_lj, nx, nz);

    // Rotation 6: align azimuth to the requested reference phase.
    let phi_n = ny.atan2(nx);
    let angle6 = std::f64::consts::FRAC_PI_2 - phi_n - phase;
    (s1x, s1y) = rotate_z(angle6, s1x, s1y);
    (s2x, s2y) = rotate_z(angle6, s2x, s2y);

    (
        iota,
        a1 * s1x,
        a1 * s1y,
        a1 * s1z,
        a2 * s2x,
        a2 * s2y,
        a2 * s2z,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Approximate solar mass in kilograms (IAU 2015 nominal).
    const MSUN_KG: f64 = 1.988_416e30;

    fn solar(m: f64) -> Mass {
        Mass::new::<kilogram>(m * MSUN_KG)
    }

    // ── total_mass ────────────────────────────────────────────────────────────

    #[test]
    fn total_mass_basic() {
        let m = total_mass(solar(30.0), solar(20.0));
        assert!((m.get::<kilogram>() / MSUN_KG - 50.0).abs() < 1e-10);
    }

    #[test]
    fn total_mass_equal() {
        let m = total_mass(solar(15.0), solar(15.0));
        assert!((m.get::<kilogram>() / MSUN_KG - 30.0).abs() < 1e-10);
    }

    // ── mass_ratio ────────────────────────────────────────────────────────────

    #[test]
    fn mass_ratio_half() {
        let q = mass_ratio(solar(30.0), solar(15.0));
        assert!((q - 0.5).abs() < 1e-10);
    }

    #[test]
    fn mass_ratio_equal_is_one() {
        let q = mass_ratio(solar(20.0), solar(20.0));
        assert!((q - 1.0).abs() < 1e-10);
    }

    // ── symmetric_mass_ratio ──────────────────────────────────────────────────

    #[test]
    fn eta_equal_mass_is_quarter() {
        let eta = symmetric_mass_ratio(solar(30.0), solar(30.0));
        assert!((eta - 0.25).abs() < 1e-10);
    }

    #[test]
    fn eta_never_exceeds_quarter() {
        for (m1, m2) in [(10.0, 5.0), (100.0, 1.0), (50.0, 50.0), (3.0, 1.0)] {
            let eta = symmetric_mass_ratio(solar(m1), solar(m2));
            assert!(eta <= 0.25 + 1e-12, "eta={eta} for m1={m1} m2={m2}");
            assert!(eta > 0.0, "eta must be positive for m1={m1} m2={m2}");
        }
    }

    // ── chirp_mass ────────────────────────────────────────────────────────────

    #[test]
    fn chirp_mass_equal_masses() {
        let m = 30.0_f64;
        let mc = chirp_mass(solar(m), solar(m));
        // Mc = M * eta^(3/5) = 2m * (1/4)^(3/5)
        let expected_msun = 2.0 * m * 0.25_f64.powf(3.0 / 5.0);
        let mc_msun = mc.get::<kilogram>() / MSUN_KG;
        // Use relative tolerance: floating-point errors scale with the magnitude
        assert!(
            (mc_msun - expected_msun).abs() / expected_msun < 1e-10,
            "mc_msun={mc_msun} expected={expected_msun}"
        );
    }

    #[test]
    fn chirp_mass_never_exceeds_total() {
        for (m1, m2) in [(30.0, 30.0), (30.0, 10.0), (100.0, 1.0)] {
            let mc = chirp_mass(solar(m1), solar(m2));
            let mt = total_mass(solar(m1), solar(m2));
            assert!(
                mc.get::<kilogram>() <= mt.get::<kilogram>() + 1e-6,
                "Mc > M for m1={m1} m2={m2}"
            );
        }
    }

    // ── chi_eff ───────────────────────────────────────────────────────────────

    #[test]
    fn chi_eff_aligned() {
        // Both spins fully aligned: chi_eff = a (for equal masses)
        let x = chi_eff(solar(30.0), solar(30.0), 0.5, 0.5, 0.0, 0.0);
        assert!((x - 0.5).abs() < 1e-10);
    }

    #[test]
    fn chi_eff_anti_aligned() {
        let x = chi_eff(
            solar(30.0),
            solar(30.0),
            0.5,
            0.5,
            std::f64::consts::PI,
            std::f64::consts::PI,
        );
        assert!((x - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn chi_eff_zero_spins() {
        let x = chi_eff(solar(30.0), solar(10.0), 0.0, 0.0, 0.0, 0.0);
        assert!(x.abs() < 1e-10);
    }

    // ── chi_p ─────────────────────────────────────────────────────────────────

    #[test]
    fn chi_p_in_plane_primary() {
        let x = chi_p(
            solar(30.0),
            solar(30.0),
            1.0,
            0.0,
            std::f64::consts::FRAC_PI_2,
            0.0,
        );
        assert!((x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn chi_p_aligned_spins_is_zero() {
        let x = chi_p(solar(30.0), solar(30.0), 1.0, 1.0, 0.0, 0.0);
        assert!(x.abs() < 1e-10);
    }

    // ── masses_from_chirp_mass_q ──────────────────────────────────────────────

    #[test]
    fn masses_from_mc_q_roundtrip_equal() {
        let m1 = solar(30.0);
        let m2 = solar(30.0);
        let (r1, r2) = masses_from_chirp_mass_q(chirp_mass(m1, m2), mass_ratio(m1, m2));
        assert!((r1.get::<kilogram>() - m1.get::<kilogram>()).abs() / m1.get::<kilogram>() < 1e-10);
        assert!((r2.get::<kilogram>() - m2.get::<kilogram>()).abs() / m2.get::<kilogram>() < 1e-10);
    }

    #[test]
    fn masses_from_mc_q_roundtrip_asymmetric() {
        let m1 = solar(36.0);
        let m2 = solar(12.0);
        let (r1, r2) = masses_from_chirp_mass_q(chirp_mass(m1, m2), mass_ratio(m1, m2));
        assert!((r1.get::<kilogram>() - m1.get::<kilogram>()).abs() / m1.get::<kilogram>() < 1e-10);
        assert!((r2.get::<kilogram>() - m2.get::<kilogram>()).abs() / m2.get::<kilogram>() < 1e-10);
    }

    #[test]
    fn masses_from_mc_q_ordering() {
        // m1 >= m2 must hold for any q in (0, 1]
        let (m1, m2) = masses_from_chirp_mass_q(solar(26.0), 0.3);
        assert!(m1.get::<kilogram>() >= m2.get::<kilogram>());
    }

    // ── masses_from_chirp_mass_eta ────────────────────────────────────────────

    #[test]
    fn masses_from_mc_eta_roundtrip_equal() {
        let m1 = solar(30.0);
        let m2 = solar(30.0);
        let mc = chirp_mass(m1, m2);
        let eta = symmetric_mass_ratio(m1, m2);
        let (r1, r2) = masses_from_chirp_mass_eta(mc, eta);
        assert!((r1.get::<kilogram>() - m1.get::<kilogram>()).abs() / m1.get::<kilogram>() < 1e-10);
        assert!((r2.get::<kilogram>() - m2.get::<kilogram>()).abs() / m2.get::<kilogram>() < 1e-10);
    }

    #[test]
    fn masses_from_mc_eta_roundtrip_asymmetric() {
        let m1 = solar(40.0);
        let m2 = solar(10.0);
        let mc = chirp_mass(m1, m2);
        let eta = symmetric_mass_ratio(m1, m2);
        let (r1, r2) = masses_from_chirp_mass_eta(mc, eta);
        assert!((r1.get::<kilogram>() - m1.get::<kilogram>()).abs() / m1.get::<kilogram>() < 1e-10);
        assert!((r2.get::<kilogram>() - m2.get::<kilogram>()).abs() / m2.get::<kilogram>() < 1e-10);
    }

    #[test]
    fn masses_from_mc_eta_ordering() {
        let (m1, m2) = masses_from_chirp_mass_eta(solar(20.0), 0.18);
        assert!(m1.get::<kilogram>() >= m2.get::<kilogram>());
    }

    // ── spin_components ───────────────────────────────────────────────────────

    #[test]
    fn spin_components_aligned() {
        let (s1x, s1y, s1z, s2x, s2y, s2z) = spin_components(0.5, 0.3, 0.0, 0.0, 0.0);
        assert!(s1x.abs() < 1e-14, "S1x={s1x}");
        assert!(s1y.abs() < 1e-14, "S1y={s1y}");
        assert!((s1z - 0.5).abs() < 1e-14, "S1z={s1z}");
        assert!(s2x.abs() < 1e-14, "S2x={s2x}");
        assert!(s2y.abs() < 1e-14, "S2y={s2y}");
        assert!((s2z - 0.3).abs() < 1e-14, "S2z={s2z}");
    }

    #[test]
    fn spin_components_antialigned_s1() {
        let (_, _, s1z, _, _, _) = spin_components(0.6, 0.0, std::f64::consts::PI, 0.0, 0.0);
        assert!((s1z - (-0.6)).abs() < 1e-14, "S1z={s1z}");
    }

    #[test]
    fn spin_components_in_plane_s1() {
        let (s1x, s1y, s1z, _, _, _) =
            spin_components(0.8, 0.0, std::f64::consts::FRAC_PI_2, 0.0, 0.0);
        assert!((s1x - 0.8).abs() < 1e-14, "S1x={s1x}");
        assert!(s1y.abs() < 1e-14, "S1y={s1y}");
        assert!(s1z.abs() < 1e-14, "S1z={s1z}");
    }

    #[test]
    fn spin_components_s1y_always_zero() {
        for (a1, t1, phi) in [(0.5, 0.3, 1.2), (0.9, 2.1, 0.0), (0.0, 1.0, 3.0)] {
            let (_, s1y, _, _, _, _) = spin_components(a1, 0.0, t1, 0.0, phi);
            assert!(s1y.abs() < 1e-14, "S1y={s1y} for a1={a1} t1={t1} phi={phi}");
        }
    }

    #[test]
    fn spin_components_s2_phi12_quarter_turn() {
        // tilt2 = π/2, phi12 = π/2 → S2 along y
        let (_, _, _, s2x, s2y, s2z) = spin_components(
            0.0,
            0.5,
            0.0,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::FRAC_PI_2,
        );
        assert!(s2x.abs() < 1e-14, "S2x={s2x}");
        assert!((s2y - 0.5).abs() < 1e-14, "S2y={s2y}");
        assert!(s2z.abs() < 1e-14, "S2z={s2z}");
    }

    // ── orbital_angular_momentum ─────────────────────────────────────────────

    #[test]
    fn oam_positive() {
        let l = orbital_angular_momentum(solar(30.0), solar(20.0), 20.0);
        assert!(l > 0.0);
    }

    #[test]
    fn oam_newtonian_formula() {
        // |L_N| = μ (G M)^{2/3} / (π f)^{1/3}
        let m1 = 30.0 * MSUN_KG;
        let m2 = 20.0 * MSUN_KG;
        let f = 20.0_f64;
        let m = m1 + m2;
        let mu = m1 * m2 / m;
        let expected = mu * (G_SI * m).powf(2.0 / 3.0) / (std::f64::consts::PI * f).powf(1.0 / 3.0);
        let got = orbital_angular_momentum(solar(30.0), solar(20.0), f);
        assert!(
            (got / expected - 1.0).abs() < 1e-10,
            "got={got} expected={expected}"
        );
    }

    #[test]
    fn oam_scales_as_f_minus_one_third() {
        // L(f) / L(8f) = 2
        let m1 = solar(30.0);
        let m2 = solar(30.0);
        let l_lo = orbital_angular_momentum(m1, m2, 20.0);
        let l_hi = orbital_angular_momentum(m1, m2, 160.0);
        assert!((l_lo / l_hi - 2.0).abs() < 1e-10);
    }

    #[test]
    fn oam_symmetric_in_masses() {
        let l_ab = orbital_angular_momentum(solar(30.0), solar(20.0), 20.0);
        let l_ba = orbital_angular_momentum(solar(20.0), solar(30.0), 20.0);
        assert!((l_ab / l_ba - 1.0).abs() < 1e-12);
    }

    // ── transform_precessing_spins ───────────────────────────────────────────
    //
    // Reference values below were generated with the actual LALSimulation
    // C library (`lalsimulation.SimInspiralTransformPrecessingNewInitialConditions`,
    // via its Python/SWIG bindings), not transcribed from documentation, and
    // agree with this Rust port to within 1e-9 (bit-for-bit, in practice) —
    // confirming the port is numerically faithful to upstream LALSuite.

    /// Solar mass in kilograms, matching LALSuite's `LAL_MSUN_SI` exactly —
    /// needed here (unlike `solar()`) because these reference values were
    /// generated with that precise constant.
    const LAL_MSUN_KG: f64 = 1.988_409_870_698_050_731_911_960_804_878_414_216e30;

    fn lal_solar(m: f64) -> Mass {
        Mass::new::<kilogram>(m * LAL_MSUN_KG)
    }

    #[test]
    fn transform_precessing_spins_matches_lalsimulation_precessing() {
        let (iota, s1x, s1y, s1z, s2x, s2y, s2z) = transform_precessing_spins(
            0.4,
            0.3,
            0.5,
            0.3,
            1.2,
            0.6,
            0.4,
            lal_solar(30.0),
            lal_solar(20.0),
            20.0,
            0.0,
        );
        let expected = [
            0.383072563130684,
            -0.28710948916277035,
            0.01771231708241292,
            0.5265495371342236,
            -0.04953631276943223,
            -0.10732802301558862,
            0.38213459565024244,
        ];
        let got = [iota, s1x, s1y, s1z, s2x, s2y, s2z];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-9, "got={got:?} expected={expected:?}");
        }
    }

    #[test]
    fn transform_precessing_spins_matches_lalsimulation_nonzero_phase() {
        // Same system as above but with a nonzero reference phase — this
        // exercises the phase-dependent azimuth handling that differs from
        // `spin_components` (which fixes S1's azimuth at zero).
        let (iota, s1x, s1y, s1z, s2x, s2y, s2z) = transform_precessing_spins(
            0.4,
            0.3,
            0.5,
            0.3,
            1.2,
            0.6,
            0.4,
            lal_solar(30.0),
            lal_solar(20.0),
            20.0,
            0.7,
        );
        let expected = [
            0.38307256313068433,
            -0.2081828617349333,
            0.19850813843162324,
            0.5265495371342236,
            -0.10703007257147708,
            -0.05017683103355657,
            0.38213459565024244,
        ];
        let got = [iota, s1x, s1y, s1z, s2x, s2y, s2z];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-9, "got={got:?} expected={expected:?}");
        }
    }

    #[test]
    fn transform_precessing_spins_matches_lalsimulation_extreme_mass_ratio() {
        let (iota, s1x, s1y, s1z, s2x, s2y, s2z) = transform_precessing_spins(
            2.7,
            5.0,
            0.1,
            3.0,
            0.2,
            0.99,
            0.99,
            lal_solar(40.0),
            lal_solar(35.0),
            10.0,
            1.5,
        );
        let expected = [
            2.7443427707016905,
            0.0948327860585488,
            0.02784090905974611,
            0.9850541236252456,
            0.12356067828990265,
            0.0652020690433003,
            -0.980092571634441,
        ];
        let got = [iota, s1x, s1y, s1z, s2x, s2y, s2z];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-9, "got={got:?} expected={expected:?}");
        }
    }

    #[test]
    fn transform_precessing_spins_matches_lalsimulation_antialigned_s1() {
        let (iota, s1x, s1y, s1z, s2x, s2y, s2z) = transform_precessing_spins(
            3.14159,
            1.0,
            std::f64::consts::PI,
            0.0,
            0.5,
            0.5,
            0.5,
            lal_solar(30.0),
            lal_solar(20.0),
            20.0,
            0.0,
        );
        let expected = [3.141590000011358, 0.0, 0.0, -0.5, 0.0, 0.0, 0.5];
        let got = [iota, s1x, s1y, s1z, s2x, s2y, s2z];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-9, "got={got:?} expected={expected:?}");
        }
    }

    #[test]
    fn transform_precessing_spins_aligned_reduces_to_theta_jn() {
        let (iota, s1x, s1y, s1z, s2x, s2y, s2z) = transform_precessing_spins(
            0.4,
            0.3,
            0.0,
            0.0,
            1.2,
            0.6,
            0.4,
            solar(30.0),
            solar(20.0),
            20.0,
            0.0,
        );
        assert!((iota - 0.4).abs() < 1e-12);
        assert!(s1x.abs() < 1e-12 && s1y.abs() < 1e-12 && (s1z - 0.6).abs() < 1e-12);
        assert!(s2x.abs() < 1e-12 && s2y.abs() < 1e-12 && (s2z - 0.4).abs() < 1e-12);
    }

    // ── Property tests ────────────────────────────────────────────────────────

    proptest! {
        #[test]
        fn prop_eta_in_range(
            m1 in 1.0_f64..200.0,
            m2 in 1.0_f64..200.0
        ) {
            let eta = symmetric_mass_ratio(solar(m1), solar(m2));
            prop_assert!(eta > 0.0 && eta <= 0.25 + 1e-12);
        }

        #[test]
        fn prop_chirp_mass_le_total(
            m1 in 1.0_f64..200.0,
            m2 in 1.0_f64..200.0
        ) {
            let mc = chirp_mass(solar(m1), solar(m2)).get::<kilogram>();
            let mt = total_mass(solar(m1), solar(m2)).get::<kilogram>();
            prop_assert!(mc <= mt + 1e-6);
        }

        #[test]
        fn prop_mass_ratio_in_range(
            m1 in 1.0_f64..200.0,
            m2 in 0.01_f64..200.0
        ) {
            // q is m2/m1, so no ordering constraint; just check it's positive
            let q = mass_ratio(solar(m1), solar(m2));
            prop_assert!(q > 0.0);
        }

        #[test]
        fn prop_chi_eff_in_range(
            m1 in 1.0_f64..200.0,
            m2 in 1.0_f64..200.0,
            a1 in 0.0_f64..=1.0,
            a2 in 0.0_f64..=1.0,
            tilt1 in 0.0_f64..=std::f64::consts::PI,
            tilt2 in 0.0_f64..=std::f64::consts::PI,
        ) {
            let x = chi_eff(solar(m1), solar(m2), a1, a2, tilt1, tilt2);
            prop_assert!(x >= -1.0 - 1e-10 && x <= 1.0 + 1e-10);
        }

        #[test]
        fn prop_chi_p_in_range(
            // Enforce m1 >= m2 as required by the chi_p definition
            m2 in 1.0_f64..200.0,
            dm in 0.0_f64..200.0,
            a1 in 0.0_f64..=1.0,
            a2 in 0.0_f64..=1.0,
            tilt1 in 0.0_f64..=std::f64::consts::PI,
            tilt2 in 0.0_f64..=std::f64::consts::PI,
        ) {
            let m1 = m2 + dm; // guarantees m1 >= m2
            let x = chi_p(solar(m1), solar(m2), a1, a2, tilt1, tilt2);
            prop_assert!(x >= 0.0 - 1e-10 && x <= 1.0 + 1e-10,
                "chi_p={x} out of [0,1] for m1={m1} m2={m2} a1={a1} a2={a2} tilt1={tilt1} tilt2={tilt2}");
        }

        #[test]
        fn prop_masses_from_mc_q_roundtrip(
            m1 in 1.0_f64..200.0,
            m2 in 1.0_f64..200.0,
        ) {
            prop_assume!(m1 >= m2);
            let mc = chirp_mass(solar(m1), solar(m2));
            let q  = mass_ratio(solar(m1), solar(m2));
            let (r1, r2) = masses_from_chirp_mass_q(mc, q);
            prop_assert!((r1.get::<kilogram>() / (m1 * MSUN_KG) - 1.0).abs() < 1e-9,
                "m1 roundtrip failed: got {} expected {}", r1.get::<kilogram>() / MSUN_KG, m1);
            prop_assert!((r2.get::<kilogram>() / (m2 * MSUN_KG) - 1.0).abs() < 1e-9,
                "m2 roundtrip failed: got {} expected {}", r2.get::<kilogram>() / MSUN_KG, m2);
        }

        #[test]
        fn prop_spin_components_s1_magnitude(
            a1 in 0.0_f64..=1.0,
            tilt1 in 0.0_f64..=std::f64::consts::PI,
            phi12 in 0.0_f64..=(2.0 * std::f64::consts::PI),
        ) {
            let (s1x, s1y, s1z, _, _, _) = spin_components(a1, 0.0, tilt1, 0.0, phi12);
            let mag2 = s1x*s1x + s1y*s1y + s1z*s1z;
            prop_assert!((mag2 - a1*a1).abs() < 1e-12,
                "|S1|²={mag2} ≠ a1²={} for a1={a1} tilt1={tilt1}", a1*a1);
        }

        #[test]
        fn prop_spin_components_s2_magnitude(
            a2 in 0.0_f64..=1.0,
            tilt2 in 0.0_f64..=std::f64::consts::PI,
            phi12 in 0.0_f64..=(2.0 * std::f64::consts::PI),
        ) {
            let (_, _, _, s2x, s2y, s2z) = spin_components(0.0, a2, 0.0, tilt2, phi12);
            let mag2 = s2x*s2x + s2y*s2y + s2z*s2z;
            prop_assert!((mag2 - a2*a2).abs() < 1e-12,
                "|S2|²={mag2} ≠ a2²={} for a2={a2} tilt2={tilt2} phi12={phi12}", a2*a2);
        }

        #[test]
        fn prop_spin_components_s1y_zero(
            a1 in 0.0_f64..=1.0,
            tilt1 in 0.0_f64..=std::f64::consts::PI,
            phi12 in 0.0_f64..=(2.0 * std::f64::consts::PI),
        ) {
            let (_, s1y, _, _, _, _) = spin_components(a1, 0.0, tilt1, 0.0, phi12);
            prop_assert!(s1y.abs() < 1e-14, "S1y={s1y} ≠ 0 for a1={a1} tilt1={tilt1}");
        }

        #[test]
        fn prop_oam_positive(
            m1 in 1.0_f64..200.0,
            m2 in 1.0_f64..200.0,
            f_ref in 1.0_f64..200.0,
        ) {
            let l = orbital_angular_momentum(solar(m1), solar(m2), f_ref);
            prop_assert!(l > 0.0, "L={l} not positive for m1={m1} m2={m2} f={f_ref}");
        }

        #[test]
        fn prop_oam_symmetric(
            m1 in 1.0_f64..200.0,
            m2 in 1.0_f64..200.0,
            f_ref in 1.0_f64..200.0,
        ) {
            let l_ab = orbital_angular_momentum(solar(m1), solar(m2), f_ref);
            let l_ba = orbital_angular_momentum(solar(m2), solar(m1), f_ref);
            prop_assert!((l_ab / l_ba - 1.0).abs() < 1e-10,
                "L not symmetric: L(m1,m2)={l_ab} L(m2,m1)={l_ba}");
        }

        #[test]
        fn prop_masses_from_mc_eta_roundtrip(
            m1 in 1.0_f64..200.0,
            m2 in 1.0_f64..200.0,
        ) {
            prop_assume!(m1 >= m2);
            let mc  = chirp_mass(solar(m1), solar(m2));
            let eta = symmetric_mass_ratio(solar(m1), solar(m2));
            let (r1, r2) = masses_from_chirp_mass_eta(mc, eta);
            prop_assert!((r1.get::<kilogram>() / (m1 * MSUN_KG) - 1.0).abs() < 1e-9,
                "m1 roundtrip failed: got {} expected {}", r1.get::<kilogram>() / MSUN_KG, m1);
            prop_assert!((r2.get::<kilogram>() / (m2 * MSUN_KG) - 1.0).abs() < 1e-9,
                "m2 roundtrip failed: got {} expected {}", r2.get::<kilogram>() / MSUN_KG, m2);
        }

        #[test]
        fn prop_transform_precessing_spins_s1_magnitude(
            m1 in 1.0_f64..200.0,
            m2 in 1.0_f64..200.0,
            theta_jn in 0.0_f64..std::f64::consts::PI,
            phi_jl in 0.0_f64..(2.0 * std::f64::consts::PI),
            tilt1 in 0.0_f64..=std::f64::consts::PI,
            tilt2 in 0.0_f64..=std::f64::consts::PI,
            phi12 in 0.0_f64..(2.0 * std::f64::consts::PI),
            a1 in 0.0_f64..=1.0,
            a2 in 0.0_f64..=1.0,
            f_ref in 1.0_f64..500.0,
            phase in 0.0_f64..(2.0 * std::f64::consts::PI),
        ) {
            let (_, s1x, s1y, s1z, s2x, s2y, s2z) = transform_precessing_spins(
                theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2,
                solar(m1), solar(m2), f_ref, phase,
            );
            let mag1 = (s1x * s1x + s1y * s1y + s1z * s1z).sqrt();
            let mag2 = (s2x * s2x + s2y * s2y + s2z * s2z).sqrt();
            prop_assert!((mag1 - a1).abs() < 1e-9, "|S1|={mag1} != a1={a1}");
            prop_assert!((mag2 - a2).abs() < 1e-9, "|S2|={mag2} != a2={a2}");
        }

        #[test]
        fn prop_transform_precessing_spins_aligned_matches_theta_jn(
            m1 in 1.0_f64..200.0,
            m2 in 1.0_f64..200.0,
            theta_jn in 0.0_f64..std::f64::consts::PI,
            phi_jl in 0.0_f64..(2.0 * std::f64::consts::PI),
            phi12 in 0.0_f64..(2.0 * std::f64::consts::PI),
            a1 in 0.0_f64..=1.0,
            a2 in 0.0_f64..=1.0,
            f_ref in 1.0_f64..500.0,
            phase in 0.0_f64..(2.0 * std::f64::consts::PI),
        ) {
            // When both spins are aligned with L_N (tilt=0), J is parallel to
            // L_N, so iota must equal theta_jn exactly and the transform must
            // agree with `spin_components`.
            let (iota, s1x, s1y, s1z, s2x, s2y, s2z) = transform_precessing_spins(
                theta_jn, phi_jl, 0.0, 0.0, phi12, a1, a2,
                solar(m1), solar(m2), f_ref, phase,
            );
            let (e1x, e1y, e1z, e2x, e2y, e2z) = spin_components(a1, a2, 0.0, 0.0, phi12);
            prop_assert!((iota - theta_jn).abs() < 1e-9, "iota={iota} != theta_jn={theta_jn}");
            prop_assert!((s1x - e1x).abs() < 1e-9 && (s1y - e1y).abs() < 1e-9 && (s1z - e1z).abs() < 1e-9);
            prop_assert!((s2x - e2x).abs() < 1e-9 && (s2y - e2y).abs() < 1e-9 && (s2z - e2z).abs() < 1e-9);
        }
    }
}
