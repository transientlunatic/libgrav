"""Tests for libgrav.lalsim — LALSimulation-compatible spin frame transformation.

libgrav.lalsim has no LALSuite dependency (see its module docstring), so
these tests run without lalsimulation/lal installed.  The one exception is
``test_precessing_matches_lalsim_directly``, which cross-checks against a
live LALSimulation call and is skipped if lalsimulation/lal aren't available.

Run with: pytest bindings/python/tests/test_lalsim.py
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from libgrav import lalsim as libgrav_lalsim

# Exact LAL_MSUN_SI value, so results are directly comparable to real
# LALSimulation output without requiring `lal` to be installed here.
MSUN_KG = 1.988_409_870_698_050_731_911_960_804_878_414_216e30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sol(m_sun: float) -> np.ndarray:
    return np.atleast_1d(np.float64(m_sun * MSUN_KG))


# ===========================================================================
# spins_to_lalsim
# ===========================================================================

class TestSpinsToLalsim:
    r"""Tests for libgrav.lalsim.spins_to_lalsim.

    A thin alias for :func:`libgrav.transform_precessing_spins`, matching
    the output convention of
    ``lalsimulation.SimInspiralTransformPrecessingNewInitialConditions``:
    returns ``(iota, S1x, S1y, S1z, S2x, S2y, S2z)`` as numpy arrays.

    Aligned-spin special case
    -------------------------
    When both spins are aligned or anti-aligned with the orbital angular
    momentum (tilt1, tilt2 ∈ {0, π}) or zero, the orbital and total angular
    momentum directions coincide (at leading order), giving:

    * ``iota = theta_jn``
    * ``S1x = S1y = 0``, ``S1z = a1 * cos(tilt1)``
    * ``S2x = S2y = 0``, ``S2z = a2 * cos(tilt2)``

    These results can be verified independently of lalsim.
    """

    # --- output structure ---------------------------------------------------

    def test_returns_seven_arrays(self):
        """Return value is a 7-tuple of numpy arrays."""
        result = libgrav_lalsim.spins_to_lalsim(
            theta_jn=np.array([0.4]), phi_jl=np.array([0.3]),
            tilt1=np.array([0.0]), tilt2=np.array([0.0]),
            phi12=np.array([0.0]),
            a1=np.array([0.5]), a2=np.array([0.3]),
            m1=_sol(30), m2=_sol(20),
            f_ref=np.array([20.0]), phase=np.array([0.0]),
        )
        assert len(result) == 7
        for arr in result:
            assert isinstance(arr, np.ndarray)

    def test_batch_output_length(self):
        """All output arrays have the same length as the inputs."""
        n = 10
        rng = np.random.default_rng(0)
        result = libgrav_lalsim.spins_to_lalsim(
            theta_jn=rng.uniform(0, math.pi, n),
            phi_jl=rng.uniform(0, 2 * math.pi, n),
            tilt1=rng.uniform(0, math.pi, n),
            tilt2=rng.uniform(0, math.pi, n),
            phi12=rng.uniform(0, 2 * math.pi, n),
            a1=rng.uniform(0, 1, n),
            a2=rng.uniform(0, 1, n),
            m1=np.full(n, 30.0 * MSUN_KG),
            m2=np.full(n, 20.0 * MSUN_KG),
            f_ref=np.full(n, 20.0),
            phase=np.zeros(n),
        )
        for arr in result:
            assert len(arr) == n

    # --- aligned-spin analytic results -------------------------------------

    def test_aligned_iota_equals_theta_jn(self):
        """For aligned spins, iota = theta_jn."""
        theta_jn = 0.7
        iota, *_ = libgrav_lalsim.spins_to_lalsim(
            theta_jn=np.array([theta_jn]), phi_jl=np.array([0.3]),
            tilt1=np.array([0.0]), tilt2=np.array([0.0]),
            phi12=np.array([0.0]),
            a1=np.array([0.5]), a2=np.array([0.3]),
            m1=_sol(30), m2=_sol(20),
            f_ref=np.array([20.0]), phase=np.array([0.0]),
        )
        assert math.isclose(iota[0], theta_jn, rel_tol=1e-12)

    def test_aligned_s1_transverse_zero(self):
        """For aligned spins, S1x = S1y = 0."""
        iota, s1x, s1y, s1z, s2x, s2y, s2z = libgrav_lalsim.spins_to_lalsim(
            theta_jn=np.array([0.4]), phi_jl=np.array([0.0]),
            tilt1=np.array([0.0]), tilt2=np.array([0.0]),
            phi12=np.array([0.0]),
            a1=np.array([0.6]), a2=np.array([0.4]),
            m1=_sol(30), m2=_sol(20),
            f_ref=np.array([20.0]), phase=np.array([0.0]),
        )
        assert abs(s1x[0]) < 1e-14
        assert abs(s1y[0]) < 1e-14

    def test_aligned_s1z_equals_a1(self):
        """For aligned spins, S1z = a1."""
        a1 = 0.6
        _, _, _, s1z, *_ = libgrav_lalsim.spins_to_lalsim(
            theta_jn=np.array([0.4]), phi_jl=np.array([0.0]),
            tilt1=np.array([0.0]), tilt2=np.array([0.0]),
            phi12=np.array([0.0]),
            a1=np.array([a1]), a2=np.array([0.3]),
            m1=_sol(30), m2=_sol(20),
            f_ref=np.array([20.0]), phase=np.array([0.0]),
        )
        assert math.isclose(s1z[0], a1, rel_tol=1e-12)

    def test_aligned_s2z_equals_a2(self):
        """For aligned spins, S2z = a2."""
        a2 = 0.4
        *_, s2x, s2y, s2z = libgrav_lalsim.spins_to_lalsim(
            theta_jn=np.array([0.4]), phi_jl=np.array([0.0]),
            tilt1=np.array([0.0]), tilt2=np.array([0.0]),
            phi12=np.array([0.0]),
            a1=np.array([0.5]), a2=np.array([a2]),
            m1=_sol(30), m2=_sol(20),
            f_ref=np.array([20.0]), phase=np.array([0.0]),
        )
        assert math.isclose(s2z[0], a2, rel_tol=1e-12)

    def test_antialigned_s1z_equals_minus_a1(self):
        """For anti-aligned S1 (tilt1=π), S1z = -a1."""
        a1 = 0.7
        _, _, _, s1z, *_ = libgrav_lalsim.spins_to_lalsim(
            theta_jn=np.array([0.4]), phi_jl=np.array([0.0]),
            tilt1=np.array([math.pi]), tilt2=np.array([0.0]),
            phi12=np.array([0.0]),
            a1=np.array([a1]), a2=np.array([0.0]),
            m1=_sol(30), m2=_sol(20),
            f_ref=np.array([20.0]), phase=np.array([0.0]),
        )
        assert math.isclose(s1z[0], -a1, rel_tol=1e-12)

    def test_zero_spins_give_all_zero_components(self):
        """a1=a2=0 gives all-zero spin components."""
        iota, s1x, s1y, s1z, s2x, s2y, s2z = libgrav_lalsim.spins_to_lalsim(
            theta_jn=np.array([0.5]), phi_jl=np.array([1.0]),
            tilt1=np.array([0.3]), tilt2=np.array([0.8]),
            phi12=np.array([2.0]),
            a1=np.array([0.0]), a2=np.array([0.0]),
            m1=_sol(30), m2=_sol(20),
            f_ref=np.array([20.0]), phase=np.array([0.0]),
        )
        for s in (s1x[0], s1y[0], s1z[0], s2x[0], s2y[0], s2z[0]):
            assert abs(s) < 1e-14

    # --- agreement with lalsimulation --------------------------------------

    def test_precessing_matches_lalsim_directly(self):
        """Precessing-spin result matches a direct lalsim call."""
        lalsimulation = pytest.importorskip("lalsimulation")

        theta_jn = 0.4
        phi_jl   = 0.3
        tilt1    = 0.5
        tilt2    = 0.3
        phi12    = 1.2
        a1       = 0.6
        a2       = 0.4
        m1       = 30.0 * MSUN_KG
        m2       = 20.0 * MSUN_KG
        f_ref    = 20.0
        phase    = 0.0

        expected = lalsimulation.SimInspiralTransformPrecessingNewInitialConditions(
            theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2, m1, m2, f_ref, phase
        )

        result = libgrav_lalsim.spins_to_lalsim(
            theta_jn=np.array([theta_jn]), phi_jl=np.array([phi_jl]),
            tilt1=np.array([tilt1]), tilt2=np.array([tilt2]),
            phi12=np.array([phi12]),
            a1=np.array([a1]), a2=np.array([a2]),
            m1=np.array([m1]), m2=np.array([m2]),
            f_ref=np.array([f_ref]), phase=np.array([phase]),
        )

        for i, (got, want) in enumerate(zip(result, expected)):
            assert math.isclose(got[0], want, rel_tol=1e-10), (
                f"Component {i}: got {got[0]}, want {want}"
            )

    def test_spin_magnitude_preserved_by_lalsim(self):
        """For precessing spins, |S1| = a1 and |S2| = a2 after transformation."""
        rng = np.random.default_rng(17)
        n = 20
        a1 = rng.uniform(0.0, 1.0, n)
        a2 = rng.uniform(0.0, 1.0, n)
        iota, s1x, s1y, s1z, s2x, s2y, s2z = libgrav_lalsim.spins_to_lalsim(
            theta_jn=rng.uniform(0, math.pi, n),
            phi_jl=rng.uniform(0, 2 * math.pi, n),
            tilt1=rng.uniform(0.01, math.pi - 0.01, n),  # avoid aligned limit
            tilt2=rng.uniform(0.01, math.pi - 0.01, n),
            phi12=rng.uniform(0, 2 * math.pi, n),
            a1=a1, a2=a2,
            m1=np.full(n, 30.0 * MSUN_KG),
            m2=np.full(n, 20.0 * MSUN_KG),
            f_ref=np.full(n, 20.0),
            phase=np.zeros(n),
        )
        np.testing.assert_allclose(s1x**2 + s1y**2 + s1z**2, a1**2, rtol=1e-10)
        np.testing.assert_allclose(s2x**2 + s2y**2 + s2z**2, a2**2, rtol=1e-10)

    def test_iota_in_zero_to_pi(self):
        """Returned iota lies in [0, π]."""
        rng = np.random.default_rng(3)
        n = 30
        iota, *_ = libgrav_lalsim.spins_to_lalsim(
            theta_jn=rng.uniform(0, math.pi, n),
            phi_jl=rng.uniform(0, 2 * math.pi, n),
            tilt1=rng.uniform(0, math.pi, n),
            tilt2=rng.uniform(0, math.pi, n),
            phi12=rng.uniform(0, 2 * math.pi, n),
            a1=rng.uniform(0, 1, n),
            a2=rng.uniform(0, 1, n),
            m1=np.full(n, 30.0 * MSUN_KG),
            m2=np.full(n, 20.0 * MSUN_KG),
            f_ref=np.full(n, 20.0),
            phase=np.zeros(n),
        )
        assert np.all(iota >= 0.0)
        assert np.all(iota <= math.pi)

    # --- unit inputs --------------------------------------------------------

    def test_astropy_mass_units(self):
        """Accepts astropy mass Quantities for m1 and m2."""
        astropy = pytest.importorskip("astropy")
        from astropy import units as u

        result_units = libgrav_lalsim.spins_to_lalsim(
            theta_jn=np.array([0.4]), phi_jl=np.array([0.3]),
            tilt1=np.array([0.0]), tilt2=np.array([0.0]),
            phi12=np.array([0.0]),
            a1=np.array([0.5]), a2=np.array([0.3]),
            m1=30 * u.Msun, m2=20 * u.Msun,
            f_ref=np.array([20.0]), phase=np.array([0.0]),
        )
        result_plain = libgrav_lalsim.spins_to_lalsim(
            theta_jn=np.array([0.4]), phi_jl=np.array([0.3]),
            tilt1=np.array([0.0]), tilt2=np.array([0.0]),
            phi12=np.array([0.0]),
            a1=np.array([0.5]), a2=np.array([0.3]),
            m1=np.array([30.0 * MSUN_KG]), m2=np.array([20.0 * MSUN_KG]),
            f_ref=np.array([20.0]), phase=np.array([0.0]),
        )
        for got, want in zip(result_units, result_plain):
            np.testing.assert_allclose(got, want, rtol=1e-3)
