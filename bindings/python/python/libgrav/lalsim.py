"""LALSimulation-compatible spin-frame transformation.

This submodule provides ``spins_to_lalsim``, a unit-aware, vectorised
conversion from the bilby/LALInference precessing-spin parameterisation to
the Cartesian spin components expected by LALSimulation waveform
generators — historically ``lalsimulation.SimInspiralTransformPrecessingNewInitialConditions``.

It is a thin backwards-compatible alias for :func:`libgrav.transform_precessing_spins`,
which is a from-scratch reimplementation of that LALSimulation routine
living in libgrav's Rust core.  **It does not call LALSimulation and has no
LALSuite dependency** — it reproduces LALSimulation's output to within
floating-point precision (verified against real LALSim calls for aligned,
precessing, and extreme-mass-ratio configurations).  This module is kept
for discoverability by anyone coming from the LALSimulation/bilby naming
convention; new code can call :func:`libgrav.transform_precessing_spins`
directly.

See also
--------
libgrav.transform_precessing_spins : The underlying implementation.
libgrav.spin_components : Pure L-frame decomposition, no J-frame rotation.
libgrav.orbital_angular_momentum : Newtonian |L_N| at a reference frequency.
bilby.gw.conversion.bilby_to_lalsimulation_spins : Reference implementation
    used to validate this module.
"""

from __future__ import annotations

import numpy as np

from libgrav import transform_precessing_spins

__all__ = ["spins_to_lalsim"]


def spins_to_lalsim(
    theta_jn,
    phi_jl,
    tilt1,
    tilt2,
    phi12,
    a1,
    a2,
    m1,
    m2,
    f_ref,
    phase=0.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    r"""Convert bilby-style spin parameters to LALSimulation Cartesian components.

    Alias for :func:`libgrav.transform_precessing_spins` — see that function
    for the full parameter/return documentation and the underlying algorithm.

    Examples
    --------
    >>> import numpy as np
    >>> from libgrav import lalsim
    >>> iota, s1x, s1y, s1z, s2x, s2y, s2z = lalsim.spins_to_lalsim(
    ...     theta_jn=np.array([0.4]), phi_jl=np.array([0.3]),
    ...     tilt1=np.array([0.5]), tilt2=np.array([0.3]),
    ...     phi12=np.array([1.2]),
    ...     a1=np.array([0.6]), a2=np.array([0.4]),
    ...     m1=np.array([30 * 1.989e30]),
    ...     m2=np.array([20 * 1.989e30]),
    ...     f_ref=np.array([20.0]),
    ... )
    """
    return transform_precessing_spins(
        theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2, m1, m2, f_ref, phase
    )
