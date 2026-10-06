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
/// use puddin::binary::total_mass;
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
/// use puddin::binary::mass_ratio;
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
/// use puddin::binary::symmetric_mass_ratio;
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
/// use puddin::binary::chirp_mass;
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
/// use puddin::binary::{chirp_mass, mass_ratio, masses_from_chirp_mass_q};
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
/// use puddin::binary::{chirp_mass, symmetric_mass_ratio, masses_from_chirp_mass_eta};
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
/// use puddin::binary::chi_eff;
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
/// use puddin::binary::chi_p;
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
    }
}
