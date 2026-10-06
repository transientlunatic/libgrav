"""libgrav — mathematical and physical primitives for gravitational-wave astronomy.

Public API
----------
All functions accept:

* plain ``float`` or ``numpy.ndarray`` (assumed SI: kg, rad, dimensionless)
* ``astropy.units.Quantity``
* ``pint.Quantity``
* ``jax.Array`` (routes automatically to the JAX-compatible backend)

Scalar inputs are broadcast to 1-D arrays.  All functions return
``numpy.ndarray`` (Rust backend) or ``jax.Array`` (JAX backend).

Mass functions return values in **kilograms**; use ``astropy.constants`` or
``pint`` to convert to solar masses.

Examples
--------
>>> import numpy as np
>>> from libgrav import chirp_mass
>>> chirp_mass(30 * 1.989e30, 30 * 1.989e30)   # plain SI
array([...])

>>> from astropy import units as u
>>> chirp_mass(30 * u.Msun, 30 * u.Msun)
array([...])
"""

from __future__ import annotations

import numpy as np

from libgrav.units import to_kg, to_rad, to_dimensionless, to_hz
from libgrav import _libgrav as _rust

# ── JAX availability detection ────────────────────────────────────────────────

try:
    import jax
    import jax.numpy as jnp
    import libgrav.jax_wrapper as _jax

    def _is_jax(x) -> bool:
        return isinstance(x, jax.Array)

    _JAX_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    _JAX_AVAILABLE = False

    def _is_jax(x) -> bool:  # type: ignore[misc]
        return False


def _any_jax(*args) -> bool:
    return any(_is_jax(a) for a in args)


# ── public API ────────────────────────────────────────────────────────────────

def total_mass(m1, m2):
    """Total mass $M = m_1 + m_2$ (kg).

    Parameters
    ----------
    m1, m2:
        Component masses.  Accepts plain SI floats/arrays, astropy Quantities,
        pint Quantities, or JAX arrays.
    """
    if _any_jax(m1, m2):
        return _jax.total_mass(m1, m2)
    return _rust.total_mass(to_kg(m1), to_kg(m2))


def mass_ratio(m1, m2):
    """Mass ratio $q = m_2 / m_1$ (dimensionless)."""
    if _any_jax(m1, m2):
        return _jax.mass_ratio(m1, m2)
    return _rust.mass_ratio(to_kg(m1), to_kg(m2))


def symmetric_mass_ratio(m1, m2):
    r"""Symmetric mass ratio $\eta = m_1 m_2 / M^2 \in (0, 1/4]$."""
    if _any_jax(m1, m2):
        return _jax.symmetric_mass_ratio(m1, m2)
    return _rust.symmetric_mass_ratio(to_kg(m1), to_kg(m2))


def chirp_mass(m1, m2):
    r"""Chirp mass $\mathcal{M} = (m_1 m_2)^{3/5} / M^{1/5}$ (kg)."""
    if _any_jax(m1, m2):
        return _jax.chirp_mass(m1, m2)
    return _rust.chirp_mass(to_kg(m1), to_kg(m2))


def masses_from_chirp_mass_q(mc, q):
    r"""Component masses $(m_1, m_2)$ from chirp mass $\mathcal{M}$ and mass ratio $q$.

    Parameters
    ----------
    mc:
        Chirp mass.  Accepts plain SI floats/arrays (kg), astropy Quantities,
        or pint Quantities.
    q:
        Mass ratio $q = m_2 / m_1 \in (0, 1]$ (dimensionless).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(m1, m2)`` in kilograms, with $m_1 \geq m_2$.
    """
    if _any_jax(mc, q):
        return _jax.masses_from_chirp_mass_q(mc, q)
    return _rust.masses_from_chirp_mass_q(to_kg(mc), to_dimensionless(q))


def masses_from_chirp_mass_eta(mc, eta):
    r"""Component masses $(m_1, m_2)$ from chirp mass $\mathcal{M}$ and symmetric mass ratio $\eta$.

    Parameters
    ----------
    mc:
        Chirp mass.  Accepts plain SI floats/arrays (kg), astropy Quantities,
        or pint Quantities.
    eta:
        Symmetric mass ratio $\eta = m_1 m_2 / M^2 \in (0, 1/4]$ (dimensionless).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(m1, m2)`` in kilograms, with $m_1 \geq m_2$.
    """
    if _any_jax(mc, eta):
        return _jax.masses_from_chirp_mass_eta(mc, eta)
    return _rust.masses_from_chirp_mass_eta(to_kg(mc), to_dimensionless(eta))


def chi_eff(m1, m2, a1, a2, tilt1, tilt2):
    r"""Effective inspiral spin $\chi_\mathrm{eff}$.

    Parameters
    ----------
    m1, m2:
        Component masses (SI or unit-carrying).
    a1, a2:
        Dimensionless spin magnitudes $\in [0, 1]$.
    tilt1, tilt2:
        Spin tilt angles (radians) w.r.t. the orbital angular momentum axis.
    """
    if _any_jax(m1, m2, a1, a2, tilt1, tilt2):
        return _jax.chi_eff(m1, m2, a1, a2, tilt1, tilt2)
    return _rust.chi_eff(
        to_kg(m1), to_kg(m2),
        to_dimensionless(a1), to_dimensionless(a2),
        to_rad(tilt1), to_rad(tilt2),
    )


def chi_p(m1, m2, a1, a2, tilt1, tilt2):
    r"""Effective precession spin $\chi_p$.

    Parameters
    ----------
    m1, m2:
        Component masses (SI or unit-carrying).
    a1, a2:
        Dimensionless spin magnitudes $\in [0, 1]$.
    tilt1, tilt2:
        Spin tilt angles (radians) w.r.t. the orbital angular momentum axis.
    """
    if _any_jax(m1, m2, a1, a2, tilt1, tilt2):
        return _jax.chi_p(m1, m2, a1, a2, tilt1, tilt2)
    return _rust.chi_p(
        to_kg(m1), to_kg(m2),
        to_dimensionless(a1), to_dimensionless(a2),
        to_rad(tilt1), to_rad(tilt2),
    )


# ── spin parameter conversions ────────────────────────────────────────────────


def spin_components(
    a1, a2, tilt1, tilt2, phi12
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    r"""Decompose spin tilts and azimuths into Cartesian components in the L-frame.

    Converts from the bilby/LALInference spin parameterisation
    (dimensionless magnitude + tilt angle + relative azimuth) to the
    Cartesian spin components used internally by LALSimulation.

    The **L-frame** has its z-axis aligned with the Newtonian orbital angular
    momentum :math:`\hat{L}`.  By convention, spin 1 is placed in the x-z
    plane (its azimuthal angle is zero), so :math:`S_{1y} = 0` identically.
    Spin 2 is rotated by :math:`\phi_{12}` around the z-axis relative to
    spin 1:

    .. math::

        \mathbf{S}_1 &= a_1 \begin{pmatrix} \sin\theta_1 \\ 0 \\ \cos\theta_1 \end{pmatrix}

        \mathbf{S}_2 &= a_2 \begin{pmatrix}
            \sin\theta_2 \cos\phi_{12} \\
            \sin\theta_2 \sin\phi_{12} \\
            \cos\theta_2
        \end{pmatrix}

    .. note::
        This function gives the spin components *in the L-frame at a fixed
        instant*.  For precessing systems, the full frame transformation
        (which also determines the inclination :math:`\iota`) requires
        knowledge of the orbital angular momentum magnitude
        :math:`|L|(f_\mathrm{ref})` and is provided by
        :func:`libgrav.lalsim.spins_to_lalsim`.

    Parameters
    ----------
    a1, a2 : array-like
        Dimensionless spin magnitudes :math:`\chi_1, \chi_2 \in [0, 1]`.
        Accepts plain floats/arrays or unit-carrying Quantities (dimensionless).
    tilt1, tilt2 : array-like
        Spin tilt angles :math:`\theta_1, \theta_2` measured from the
        orbital angular momentum axis.  Values in :math:`[0, \pi]`.
        Accepts plain floats/arrays in radians, or ``astropy``/``pint``
        angle Quantities.
    phi12 : array-like
        Azimuthal angle of spin 2 relative to spin 1 in the orbital plane.
        Values in :math:`[0, 2\pi)`.  Accepts plain floats/arrays in radians
        or angle Quantities.

    Returns
    -------
    S1x, S1y, S1z, S2x, S2y, S2z : numpy.ndarray
        Cartesian spin components in the L-frame, each of shape matching
        the broadcast shape of the inputs.  Components are dimensionless
        (:math:`|\mathbf{S}_i| = a_i`).
    """
    if _any_jax(a1, a2, tilt1, tilt2, phi12):
        return _jax.spin_components(a1, a2, tilt1, tilt2, phi12)
    return _rust.spin_components(
        to_dimensionless(a1),
        to_dimensionless(a2),
        to_rad(tilt1),
        to_rad(tilt2),
        to_rad(phi12),
    )


def orbital_angular_momentum(m1, m2, f_ref) -> np.ndarray:
    r"""Newtonian orbital angular momentum magnitude at a reference frequency.

    Computes

    .. math::

        |L_\mathrm{N}| = \mu \frac{(G M)^{2/3}}{(\pi f_\mathrm{ref})^{1/3}}

    where :math:`\mu = m_1 m_2 / M` is the reduced mass and
    :math:`M = m_1 + m_2` is the total mass.  This is the leading-order
    (Newtonian) expression for the orbital angular momentum at gravitational-
    wave frequency :math:`f_\mathrm{ref}`.

    Higher-order post-Newtonian corrections are not included; for production
    injection generation use :func:`libgrav.lalsim.spins_to_lalsim`, which
    delegates to LALSimulation's full PN implementation.

    Parameters
    ----------
    m1, m2 : array-like
        Component masses in SI (kg), or unit-carrying mass Quantities.
    f_ref : array-like
        Gravitational-wave reference frequency in Hz.  Plain floats/arrays
        are assumed to be in Hz.

    Returns
    -------
    numpy.ndarray
        :math:`|L_\mathrm{N}|` in SI units (kg m² s⁻¹).

    Notes
    -----
    The gravitational constant used is :math:`G = 6.67430 \times 10^{-11}`
    m³ kg⁻¹ s⁻², consistent with the LALSuite convention (CODATA 2014).
    """
    if _any_jax(m1, m2):
        return _jax.orbital_angular_momentum(m1, m2, f_ref)
    m1_, m2_, f_ = np.broadcast_arrays(
        to_kg(m1),
        to_kg(m2),
        to_hz(f_ref),
    )
    return _rust.orbital_angular_momentum(
        np.ascontiguousarray(m1_),
        np.ascontiguousarray(m2_),
        np.ascontiguousarray(f_),
    )


def transform_precessing_spins(
    theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2, m1, m2, f_ref, phase=0.0
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    r"""Transform precessing-spin parameters into Cartesian components and inclination.

    Converts the bilby / LALInference spin parameterisation (angles measured
    relative to the total angular momentum :math:`\mathbf{J}`) into the
    inclination and Cartesian spin components used directly by waveform
    generators, by constructing
    :math:`\mathbf{J} = \mathbf{L}_\mathrm{N} + \mathbf{S}_1 + \mathbf{S}_2`
    (using the Newtonian orbital angular momentum with its 1PN
    point-particle correction — no spin-orbit term) and rotating from the
    L-frame through the J-frame into the frame the requested reference
    orbital phase implies.

    This is a from-scratch reimplementation of LALSimulation's
    ``SimInspiralTransformPrecessingNewInitialConditions`` — it does not
    call LALSimulation and has no LALSuite dependency, so it is available
    wherever libgrav is (Python, Julia, WASM, R, the CLI).  It reproduces
    LALSimulation's output to within floating-point precision (verified
    against ``lalsimulation.SimInspiralTransformPrecessingNewInitialConditions``
    for aligned, precessing, and extreme-mass-ratio configurations).

    Parameters
    ----------
    theta_jn : array-like
        Inclination of :math:`\mathbf{J}` relative to the line of sight, in
        radians.  Accepts astropy/pint angle Quantities.
    phi_jl : array-like
        Azimuthal angle of :math:`\mathbf{L}_\mathrm{N}` about
        :math:`\mathbf{J}`, in radians.
    tilt1, tilt2 : array-like
        Spin tilt angles :math:`\theta_1, \theta_2` measured from
        :math:`\mathbf{L}_\mathrm{N}`, in radians.  Values in
        :math:`[0, \pi]`.
    phi12 : array-like
        Azimuthal angle of spin 2 relative to spin 1 in the orbital plane,
        in radians.
    a1, a2 : array-like
        Dimensionless spin magnitudes :math:`\chi_1, \chi_2 \in [0, 1]`.
    m1, m2 : array-like
        Component masses in SI (kg), or unit-carrying mass Quantities.
    f_ref : array-like
        Gravitational-wave reference frequency in Hz at which the spin
        components are defined.
    phase : array-like, optional
        Reference orbital phase in radians.  Default 0.

    Returns
    -------
    iota : numpy.ndarray
        Inclination of :math:`\mathbf{L}_\mathrm{N}` relative to the line of
        sight in radians, after the J→L rotation.
    S1x, S1y, S1z, S2x, S2y, S2z : numpy.ndarray
        Cartesian dimensionless spin components, each of the same shape as
        the broadcast inputs.

    Notes
    -----
    Like LALSimulation's routine, this omits spin-orbit corrections to
    :math:`|\mathbf{L}_\mathrm{N}|` (only the Newtonian value and its 1PN
    point-particle correction are used).  For the simpler aligned-spin-only
    decomposition with no J-frame rotation, see :func:`spin_components`.

    Examples
    --------
    >>> import numpy as np
    >>> from libgrav import transform_precessing_spins
    >>> iota, s1x, s1y, s1z, s2x, s2y, s2z = transform_precessing_spins(
    ...     theta_jn=np.array([0.4]), phi_jl=np.array([0.3]),
    ...     tilt1=np.array([0.5]), tilt2=np.array([0.3]),
    ...     phi12=np.array([1.2]),
    ...     a1=np.array([0.6]), a2=np.array([0.4]),
    ...     m1=np.array([30 * 1.989e30]),
    ...     m2=np.array([20 * 1.989e30]),
    ...     f_ref=np.array([20.0]),
    ... )
    """
    if _any_jax(theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2, m1, m2, f_ref, phase):
        return _jax.transform_precessing_spins(
            theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2, m1, m2, f_ref, phase
        )
    (
        theta_jn_, phi_jl_, tilt1_, tilt2_, phi12_,
        a1_, a2_, m1_, m2_, f_ref_, phase_,
    ) = np.broadcast_arrays(
        to_rad(theta_jn),
        to_rad(phi_jl),
        to_rad(tilt1),
        to_rad(tilt2),
        to_rad(phi12),
        to_dimensionless(a1),
        to_dimensionless(a2),
        to_kg(m1),
        to_kg(m2),
        to_hz(f_ref),
        to_rad(phase),
    )
    return _rust.transform_precessing_spins(
        np.ascontiguousarray(theta_jn_),
        np.ascontiguousarray(phi_jl_),
        np.ascontiguousarray(tilt1_),
        np.ascontiguousarray(tilt2_),
        np.ascontiguousarray(phi12_),
        np.ascontiguousarray(a1_),
        np.ascontiguousarray(a2_),
        np.ascontiguousarray(m1_),
        np.ascontiguousarray(m2_),
        np.ascontiguousarray(f_ref_),
        np.ascontiguousarray(phase_),
    )


__all__ = [
    "total_mass",
    "mass_ratio",
    "symmetric_mass_ratio",
    "chirp_mass",
    "masses_from_chirp_mass_q",
    "masses_from_chirp_mass_eta",
    "chi_eff",
    "chi_p",
    "spin_components",
    "orbital_angular_momentum",
    "transform_precessing_spins",
]
