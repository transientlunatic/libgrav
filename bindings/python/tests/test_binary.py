"""Python-level tests for the libgrav binary parameter functions.

Run with: pytest bindings/python/tests/

Exercises:
- plain numpy inputs (SI) via libgrav.binary submodule
- plain numpy inputs (SI) via top-level libgrav namespace (backward compat)
- astropy.units.Quantity inputs
- pint.Quantity inputs (if pint is installed)
- JAX inputs (if jax is installed)
- unit mismatch raises an error
- numerical correctness matches Rust reference values
"""

from __future__ import annotations

import math
import pytest
import numpy as np

import libgrav
import libgrav.binary
from conftest import MSUN_KG, sol


# ── libgrav.binary submodule tests ─────────────────────────────────────────────

class TestBinarySubmodule:
    """Verify that the libgrav.binary submodule exposes the same functions."""

    def test_submodule_accessible(self):
        assert hasattr(libgrav, "binary")

    def test_total_mass(self):
        result = libgrav.binary.total_mass(sol(30), sol(20))
        assert math.isclose(result[0] / MSUN_KG, 50.0, rel_tol=1e-8)

    def test_mass_ratio(self):
        result = libgrav.binary.mass_ratio(sol(30), sol(15))
        assert math.isclose(result[0], 0.5, rel_tol=1e-8)

    def test_symmetric_mass_ratio(self):
        result = libgrav.binary.symmetric_mass_ratio(sol(30), sol(30))
        assert math.isclose(result[0], 0.25, rel_tol=1e-8)

    def test_chirp_mass(self):
        mc = libgrav.binary.chirp_mass(sol(30), sol(30))[0] / MSUN_KG
        expected = 60.0 * 0.25 ** (3.0 / 5.0)
        assert math.isclose(mc, expected, rel_tol=1e-8)

    def test_masses_from_chirp_mass_q_roundtrip(self):
        mc = libgrav.binary.chirp_mass(sol(30), sol(20))
        q = libgrav.binary.mass_ratio(sol(30), sol(20))
        (m1, m2) = libgrav.binary.masses_from_chirp_mass_q(mc, q)
        assert math.isclose(m1[0] / MSUN_KG, 30.0, rel_tol=1e-8)
        assert math.isclose(m2[0] / MSUN_KG, 20.0, rel_tol=1e-8)

    def test_masses_from_chirp_mass_eta_roundtrip(self):
        mc = libgrav.binary.chirp_mass(sol(30), sol(20))
        eta = libgrav.binary.symmetric_mass_ratio(sol(30), sol(20))
        (m1, m2) = libgrav.binary.masses_from_chirp_mass_eta(mc, eta)
        assert math.isclose(m1[0] / MSUN_KG, 30.0, rel_tol=1e-8)
        assert math.isclose(m2[0] / MSUN_KG, 20.0, rel_tol=1e-8)

    def test_chi_eff(self):
        x = libgrav.binary.chi_eff(
            sol(30), sol(30), np.array([0.5]), np.array([0.5]),
            np.array([0.0]), np.array([0.0])
        )[0]
        assert math.isclose(x, 0.5, rel_tol=1e-8)

    def test_chi_p(self):
        x = libgrav.binary.chi_p(
            sol(30), sol(30), np.array([1.0]), np.array([0.0]),
            np.array([math.pi / 2]), np.array([0.0])
        )[0]
        assert math.isclose(x, 1.0, rel_tol=1e-8)

    def test_results_match_toplevel(self):
        """libgrav.binary.* and libgrav.* must return identical results."""
        m1, m2 = sol(35), sol(25)
        assert np.allclose(libgrav.binary.chirp_mass(m1, m2), libgrav.chirp_mass(m1, m2))
        assert np.allclose(libgrav.binary.total_mass(m1, m2), libgrav.total_mass(m1, m2))
        assert np.allclose(libgrav.binary.mass_ratio(m1, m2), libgrav.mass_ratio(m1, m2))


# ── plain numpy tests ─────────────────────────────────────────────────────────

class TestPlainSI:
    def test_total_mass(self):
        result = libgrav.total_mass(sol(30), sol(20))
        assert math.isclose(result[0] / MSUN_KG, 50.0, rel_tol=1e-8)

    def test_mass_ratio(self):
        result = libgrav.mass_ratio(sol(30), sol(15))
        assert math.isclose(result[0], 0.5, rel_tol=1e-8)

    def test_symmetric_mass_ratio_equal(self):
        result = libgrav.symmetric_mass_ratio(sol(30), sol(30))
        assert math.isclose(result[0], 0.25, rel_tol=1e-8)

    def test_symmetric_mass_ratio_upper_bound(self):
        for m1, m2 in [(10, 5), (100, 1), (50, 50)]:
            eta = libgrav.symmetric_mass_ratio(sol(m1), sol(m2))[0]
            assert eta <= 0.25 + 1e-10

    def test_chirp_mass_equal_masses(self):
        mc = libgrav.chirp_mass(sol(30), sol(30))[0] / MSUN_KG
        expected = 60.0 * 0.25 ** (3.0 / 5.0)
        assert math.isclose(mc, expected, rel_tol=1e-8)

    def test_chirp_mass_le_total(self):
        for m1, m2 in [(30, 30), (30, 10), (100, 1)]:
            mc = libgrav.chirp_mass(sol(m1), sol(m2))[0]
            mt = libgrav.total_mass(sol(m1), sol(m2))[0]
            assert mc <= mt + 1e-6

    def test_masses_from_chirp_mass_q_roundtrip(self):
        mc = libgrav.chirp_mass(sol(30), sol(20))
        q = libgrav.mass_ratio(sol(30), sol(20))
        (m1, m2) = libgrav.masses_from_chirp_mass_q(mc, q)
        assert math.isclose(m1[0] / MSUN_KG, 30.0, rel_tol=1e-8)
        assert math.isclose(m2[0] / MSUN_KG, 20.0, rel_tol=1e-8)

    def test_masses_from_chirp_mass_q_equal(self):
        mc = libgrav.chirp_mass(sol(30), sol(30))
        (m1, m2) = libgrav.masses_from_chirp_mass_q(mc, np.array([1.0]))
        assert math.isclose(m1[0] / MSUN_KG, 30.0, rel_tol=1e-8)
        assert math.isclose(m2[0] / MSUN_KG, 30.0, rel_tol=1e-8)

    def test_masses_from_chirp_mass_eta_roundtrip(self):
        mc = libgrav.chirp_mass(sol(30), sol(20))
        eta = libgrav.symmetric_mass_ratio(sol(30), sol(20))
        (m1, m2) = libgrav.masses_from_chirp_mass_eta(mc, eta)
        assert math.isclose(m1[0] / MSUN_KG, 30.0, rel_tol=1e-8)
        assert math.isclose(m2[0] / MSUN_KG, 20.0, rel_tol=1e-8)

    def test_masses_from_chirp_mass_eta_equal(self):
        mc = libgrav.chirp_mass(sol(30), sol(30))
        (m1, m2) = libgrav.masses_from_chirp_mass_eta(mc, np.array([0.25]))
        assert math.isclose(m1[0] / MSUN_KG, 30.0, rel_tol=1e-8)
        assert math.isclose(m2[0] / MSUN_KG, 30.0, rel_tol=1e-8)

    def test_chi_eff_aligned(self):
        x = libgrav.chi_eff(sol(30), sol(30), np.array([0.5]), np.array([0.5]),
                           np.array([0.0]), np.array([0.0]))[0]
        assert math.isclose(x, 0.5, rel_tol=1e-8)

    def test_chi_p_in_plane(self):
        x = libgrav.chi_p(sol(30), sol(30), np.array([1.0]), np.array([0.0]),
                         np.array([math.pi / 2]), np.array([0.0]))[0]
        assert math.isclose(x, 1.0, rel_tol=1e-8)


# ── astropy tests ─────────────────────────────────────────────────────────────

class TestAstropy:
    pytest.importorskip("astropy")

    def test_chirp_mass_with_astropy_units(self):
        from astropy import units as u
        mc_kg = libgrav.chirp_mass(30 * u.Msun, 30 * u.Msun)[0]
        msun_kg = (1 * u.Msun).to('kg').value
        mc_msun = mc_kg / msun_kg
        expected = 60.0 * 0.25 ** (3.0 / 5.0)
        assert math.isclose(mc_msun, expected, rel_tol=1e-6)

    def test_wrong_unit_raises(self):
        from astropy import units as u
        with pytest.raises(Exception):  # astropy raises UnitConversionError
            libgrav.chirp_mass(30 * u.meter, 30 * u.meter)


# ── pint tests ────────────────────────────────────────────────────────────────

class TestPint:
    pytest.importorskip("pint")

    def test_chirp_mass_with_pint_units(self):
        import pint
        ureg = pint.UnitRegistry()
        m1 = 30 * MSUN_KG * ureg.kg
        m2 = 30 * MSUN_KG * ureg.kg
        mc = libgrav.chirp_mass(m1, m2)[0] / MSUN_KG
        expected = 60.0 * 0.25 ** (3.0 / 5.0)
        assert math.isclose(mc, expected, rel_tol=1e-4)

    def test_wrong_unit_raises(self):
        import pint
        ureg = pint.UnitRegistry()
        with pytest.raises(Exception):
            libgrav.chirp_mass(30 * ureg.meter, 30 * ureg.meter)


# ── JAX tests ─────────────────────────────────────────────────────────────────

class TestJAX:
    jax = pytest.importorskip("jax")

    def test_chirp_mass_jit(self):
        import jax
        import jax.numpy as jnp

        m1 = jnp.array([30.0 * MSUN_KG])
        m2 = jnp.array([30.0 * MSUN_KG])
        mc_fn = jax.jit(libgrav.chirp_mass)
        mc = mc_fn(m1, m2)[0] / MSUN_KG
        expected = 60.0 * 0.25 ** (3.0 / 5.0)
        assert math.isclose(float(mc), expected, rel_tol=1e-4)

    def test_chirp_mass_grad(self):
        import jax
        import jax.numpy as jnp

        m1 = jnp.array([30.0 * MSUN_KG])
        m2 = jnp.array([30.0 * MSUN_KG])

        def scalar_mc(m1, m2):
            return libgrav.chirp_mass(m1, m2)[0]

        grad = jax.grad(scalar_mc)(m1, m2)
        # dMc/dm1 at equal masses = Mc * (3/(5*m1) - 1/(5*M))
        # = Mc * (3/5 - 1/10) / m1 = Mc * 1/2 / m1
        mc = libgrav.chirp_mass(m1, m2)[0]
        expected_grad = mc * (3.0 / (5.0 * m1[0]) - 1.0 / (5.0 * 2.0 * m1[0]))
        assert math.isclose(float(grad[0]), float(expected_grad), rel_tol=1e-5)
