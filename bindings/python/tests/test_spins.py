"""Tests for libgrav spin parameter conversion functions.

Covers:
- libgrav.spin_components: tilt/azimuth → Cartesian in the L-frame
- libgrav.orbital_angular_momentum: Newtonian orbital angular momentum magnitude

These are pure-math functions with no LALSimulation dependency.
Run with: pytest bindings/python/tests/test_spins.py
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import libgrav
from conftest import MSUN_KG, sol as _sol

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

G_SI = 6.67430e-11  # gravitational constant [m^3 kg^-1 s^-2]


# ===========================================================================
# spin_components
# ===========================================================================

class TestSpinComponents:
    """Tests for libgrav.spin_components.

    Convention
    ----------
    In the L-frame (z-axis = orbital angular momentum L):
    - Spin 1 lies in the x-z plane: S1y = 0 always.
    - Spin 2 is rotated by phi12 around the z-axis relative to spin 1.

    S1 = a1 * (sin(tilt1),  0,          cos(tilt1))
    S2 = a2 * (sin(tilt2)*cos(phi12),
               sin(tilt2)*sin(phi12),
               cos(tilt2))
    """

    # --- alignment / anti-alignment -----------------------------------------

    def test_aligned_s1z_equals_a1(self):
        """Aligned S1 (tilt1=0) gives S1z = a1."""
        s1x, s1y, s1z, *_ = libgrav.spin_components(
            a1=np.array([0.5]), a2=np.array([0.0]),
            tilt1=np.array([0.0]), tilt2=np.array([0.0]),
            phi12=np.array([0.0]),
        )
        assert math.isclose(s1z[0], 0.5, rel_tol=1e-12)

    def test_aligned_s2z_equals_a2(self):
        """Aligned S2 (tilt2=0) gives S2z = a2."""
        *_, s2x, s2y, s2z = libgrav.spin_components(
            a1=np.array([0.0]), a2=np.array([0.3]),
            tilt1=np.array([0.0]), tilt2=np.array([0.0]),
            phi12=np.array([0.0]),
        )
        assert math.isclose(s2z[0], 0.3, rel_tol=1e-12)

    def test_aligned_transverse_components_zero(self):
        """Aligned spins have zero transverse (x, y) components."""
        s1x, s1y, s1z, s2x, s2y, s2z = libgrav.spin_components(
            a1=np.array([0.7]), a2=np.array([0.4]),
            tilt1=np.array([0.0]), tilt2=np.array([0.0]),
            phi12=np.array([0.0]),
        )
        for component in (s1x[0], s1y[0], s2x[0], s2y[0]):
            assert abs(component) < 1e-14, f"Expected zero, got {component}"

    def test_antialigned_s1_gives_negative_s1z(self):
        """Anti-aligned S1 (tilt1=π) gives S1z = -a1."""
        s1x, s1y, s1z, *_ = libgrav.spin_components(
            a1=np.array([0.6]), a2=np.array([0.0]),
            tilt1=np.array([math.pi]), tilt2=np.array([0.0]),
            phi12=np.array([0.0]),
        )
        assert math.isclose(s1z[0], -0.6, rel_tol=1e-12)

    # --- in-plane spins -----------------------------------------------------

    def test_in_plane_s1_gives_s1x_equals_a1(self):
        """In-plane S1 (tilt1=π/2) gives S1x = a1."""
        s1x, s1y, s1z, *_ = libgrav.spin_components(
            a1=np.array([0.8]), a2=np.array([0.0]),
            tilt1=np.array([math.pi / 2]), tilt2=np.array([0.0]),
            phi12=np.array([0.0]),
        )
        assert math.isclose(s1x[0], 0.8, rel_tol=1e-12)
        assert abs(s1y[0]) < 1e-14
        assert abs(s1z[0]) < 1e-14

    def test_in_plane_s2_phi12_quarter_turn(self):
        """In-plane S2 with phi12=π/2 lies along the y-axis."""
        *_, s2x, s2y, s2z = libgrav.spin_components(
            a1=np.array([0.0]), a2=np.array([0.5]),
            tilt1=np.array([0.0]), tilt2=np.array([math.pi / 2]),
            phi12=np.array([math.pi / 2]),
        )
        assert abs(s2x[0]) < 1e-14
        assert math.isclose(s2y[0], 0.5, rel_tol=1e-12)
        assert abs(s2z[0]) < 1e-14

    def test_in_plane_s2_phi12_half_turn(self):
        """In-plane S2 with phi12=π points along negative x-axis."""
        *_, s2x, s2y, s2z = libgrav.spin_components(
            a1=np.array([0.0]), a2=np.array([0.4]),
            tilt1=np.array([0.0]), tilt2=np.array([math.pi / 2]),
            phi12=np.array([math.pi]),
        )
        assert math.isclose(s2x[0], -0.4, rel_tol=1e-12)
        assert abs(s2y[0]) < 1e-14
        assert abs(s2z[0]) < 1e-14

    # --- convention: S1 always in x-z plane --------------------------------

    def test_s1y_is_always_zero(self):
        """S1y is identically zero for any tilt1 and phi12."""
        rng = np.random.default_rng(99)
        n = 100
        a1    = rng.uniform(0.0, 1.0, n)
        tilt1 = rng.uniform(0.0, math.pi, n)
        phi12 = rng.uniform(0.0, 2 * math.pi, n)
        _, s1y, *_ = libgrav.spin_components(
            a1=a1, a2=np.zeros(n),
            tilt1=tilt1, tilt2=np.zeros(n),
            phi12=phi12,
        )
        np.testing.assert_array_equal(s1y, 0.0)

    # --- magnitude preservation ---------------------------------------------

    def test_s1_magnitude_preserved(self):
        """|S1|² = a1² for arbitrary tilts."""
        rng = np.random.default_rng(42)
        n = 200
        a1    = rng.uniform(0.0, 1.0, n)
        tilt1 = rng.uniform(0.0, math.pi, n)
        s1x, s1y, s1z, *_ = libgrav.spin_components(
            a1=a1, a2=np.zeros(n),
            tilt1=tilt1, tilt2=np.zeros(n),
            phi12=np.zeros(n),
        )
        np.testing.assert_allclose(s1x**2 + s1y**2 + s1z**2, a1**2, rtol=1e-12)

    def test_s2_magnitude_preserved(self):
        """|S2|² = a2² for arbitrary tilts and phi12."""
        rng = np.random.default_rng(7)
        n = 200
        a2    = rng.uniform(0.0, 1.0, n)
        tilt2 = rng.uniform(0.0, math.pi, n)
        phi12 = rng.uniform(0.0, 2 * math.pi, n)
        *_, s2x, s2y, s2z = libgrav.spin_components(
            a1=np.zeros(n), a2=a2,
            tilt1=np.zeros(n), tilt2=tilt2,
            phi12=phi12,
        )
        np.testing.assert_allclose(s2x**2 + s2y**2 + s2z**2, a2**2, rtol=1e-12)

    # --- zero spins ---------------------------------------------------------

    def test_zero_a1_gives_all_zero_s1(self):
        """a1=0 gives zero S1 components regardless of tilt1."""
        s1x, s1y, s1z, *_ = libgrav.spin_components(
            a1=np.array([0.0]), a2=np.array([0.5]),
            tilt1=np.array([1.2]), tilt2=np.array([0.8]),
            phi12=np.array([0.5]),
        )
        assert s1x[0] == 0.0
        assert s1y[0] == 0.0
        assert s1z[0] == 0.0

    def test_zero_a2_gives_all_zero_s2(self):
        """a2=0 gives zero S2 components regardless of tilt2 and phi12."""
        *_, s2x, s2y, s2z = libgrav.spin_components(
            a1=np.array([0.5]), a2=np.array([0.0]),
            tilt1=np.array([0.3]), tilt2=np.array([2.1]),
            phi12=np.array([3.0]),
        )
        assert s2x[0] == 0.0
        assert s2y[0] == 0.0
        assert s2z[0] == 0.0

    # --- output shape -------------------------------------------------------

    def test_returns_six_arrays(self):
        """Function returns a 6-tuple of arrays."""
        result = libgrav.spin_components(
            a1=np.array([0.5]), a2=np.array([0.3]),
            tilt1=np.array([0.5]), tilt2=np.array([0.3]),
            phi12=np.array([1.0]),
        )
        assert len(result) == 6
        for arr in result:
            assert isinstance(arr, np.ndarray)

    def test_batch_shape_preserved(self):
        """Output arrays have the same length as the inputs."""
        n = 50
        rng = np.random.default_rng(0)
        result = libgrav.spin_components(
            a1=rng.uniform(0, 1, n), a2=rng.uniform(0, 1, n),
            tilt1=rng.uniform(0, math.pi, n),
            tilt2=rng.uniform(0, math.pi, n),
            phi12=rng.uniform(0, 2 * math.pi, n),
        )
        for arr in result:
            assert len(arr) == n

    # --- unit inputs --------------------------------------------------------

    def test_astropy_angle_units(self):
        """Accepts astropy angle Quantities for tilt and phi12."""
        astropy = pytest.importorskip("astropy")
        from astropy import units as u

        s1x, s1y, s1z, *_ = libgrav.spin_components(
            a1=np.array([0.5]), a2=np.array([0.0]),
            tilt1=np.array([90.0]) * u.deg,
            tilt2=np.array([0.0]) * u.deg,
            phi12=np.array([0.0]) * u.deg,
        )
        assert math.isclose(s1x[0], 0.5, rel_tol=1e-12)

    # --- numerical spot-check -----------------------------------------------

    def test_45_degree_tilt_components(self):
        """45° tilt gives S1x = S1z = a1/√2."""
        a1 = 0.6
        s1x, s1y, s1z, *_ = libgrav.spin_components(
            a1=np.array([a1]), a2=np.array([0.0]),
            tilt1=np.array([math.pi / 4]), tilt2=np.array([0.0]),
            phi12=np.array([0.0]),
        )
        expected = a1 / math.sqrt(2)
        assert math.isclose(s1x[0], expected, rel_tol=1e-12)
        assert math.isclose(s1z[0], expected, rel_tol=1e-12)


# ===========================================================================
# orbital_angular_momentum
# ===========================================================================

class TestOrbitalAngularMomentum:
    r"""Tests for libgrav.orbital_angular_momentum.

    Implements the Newtonian formula:
        |L_N| = μ (G M)^{2/3} / (π f_ref)^{1/3}

    where μ = m1 m2 / M is the reduced mass, M = m1 + m2, G is Newton's
    constant, and f_ref is the gravitational-wave reference frequency.
    """

    def test_positive_for_positive_inputs(self):
        """L_N is positive for physically valid inputs."""
        L = libgrav.orbital_angular_momentum(
            m1=_sol(30), m2=_sol(20), f_ref=np.array([20.0])
        )
        assert L[0] > 0.0

    def test_matches_newtonian_formula(self):
        """Result agrees with the analytic Newtonian formula to machine precision."""
        m1 = 30.0 * MSUN_KG
        m2 = 20.0 * MSUN_KG
        f_ref = 20.0  # Hz
        M = m1 + m2
        mu = m1 * m2 / M
        expected = mu * (G_SI * M) ** (2.0 / 3.0) / (math.pi * f_ref) ** (1.0 / 3.0)

        L = libgrav.orbital_angular_momentum(
            m1=np.array([m1]), m2=np.array([m2]), f_ref=np.array([f_ref])
        )
        assert math.isclose(L[0], expected, rel_tol=1e-8)

    def test_scales_as_f_to_minus_one_third(self):
        """Doubling f_ref^3 halves L_N (L ∝ f^{-1/3} → L(f)/L(8f) = 2)."""
        m1 = _sol(30)
        m2 = _sol(30)
        L_lo = libgrav.orbital_angular_momentum(m1, m2, np.array([20.0]))
        L_hi = libgrav.orbital_angular_momentum(m1, m2, np.array([160.0]))  # 8× higher
        assert math.isclose(L_lo[0] / L_hi[0], 2.0, rel_tol=1e-10)

    def test_scales_as_m_to_five_thirds_equal_masses(self):
        """For equal masses, L_N ∝ m^{5/3}: doubling mass scales L by 2^{5/3}."""
        f_ref = np.array([20.0])
        L1 = libgrav.orbital_angular_momentum(_sol(30), _sol(30), f_ref)
        L2 = libgrav.orbital_angular_momentum(_sol(60), _sol(60), f_ref)
        ratio = L2[0] / L1[0]
        assert math.isclose(ratio, 2.0 ** (5.0 / 3.0), rel_tol=1e-8)

    def test_symmetric_in_masses(self):
        """L_N(m1, m2) = L_N(m2, m1)."""
        L_ab = libgrav.orbital_angular_momentum(_sol(30), _sol(20), np.array([20.0]))
        L_ba = libgrav.orbital_angular_momentum(_sol(20), _sol(30), np.array([20.0]))
        assert math.isclose(L_ab[0], L_ba[0], rel_tol=1e-12)

    def test_returns_array(self):
        """Output is a numpy array."""
        L = libgrav.orbital_angular_momentum(_sol(30), _sol(30), np.array([20.0]))
        assert isinstance(L, np.ndarray)

    def test_batch_inputs(self):
        """Accepts and returns arrays of multiple frequencies."""
        f_refs = np.array([10.0, 20.0, 40.0])
        L = libgrav.orbital_angular_momentum(_sol(30), _sol(30), f_refs)
        assert len(L) == 3
        # Higher frequency → smaller L
        assert L[0] > L[1] > L[2]

    def test_astropy_mass_units(self):
        """Accepts astropy mass Quantities."""
        astropy = pytest.importorskip("astropy")
        from astropy import units as u

        L = libgrav.orbital_angular_momentum(
            m1=30 * u.Msun, m2=20 * u.Msun, f_ref=np.array([20.0])
        )
        L_plain = libgrav.orbital_angular_momentum(
            m1=_sol(30), m2=_sol(20), f_ref=np.array([20.0])
        )
        # Astropy's IAU 2015 Msun (1.9884e30 kg) differs from the approximate
        # 1.989e30 kg used for _sol(), so allow 0.1% tolerance here.
        assert math.isclose(L[0], L_plain[0], rel_tol=1e-3)
