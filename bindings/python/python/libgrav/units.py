"""Unit adapter layer for libgrav.

Accepts quantities from either ``astropy.units`` or ``pint`` and normalises
them to plain SI numpy arrays.  Both libraries expose a compatible
``.to(unit).value`` / ``.to(unit).magnitude`` interface; we detect which is
present via duck-typing so callers are not forced to use a specific library.

All public functions in libgrav expect SI inputs:

* mass → kilograms
* angle → radians
* dimensionless → plain float/array
"""

from __future__ import annotations

import numpy as np

# ── SI base units as strings recognised by both astropy and pint ──────────────
_SI_UNIT = {
    "mass": "kg",
    "angle": "rad",
    "dimensionless": "",
    "frequency": "Hz",
}

# Explicit sentinel to avoid `or` mishandling zero-valued arrays.
_MISSING = object()


def _to_si(value, expected_physical_type: str) -> np.ndarray:
    """Convert *value* to a plain SI numpy array.

    Parameters
    ----------
    value:
        A plain ``float`` / ``numpy.ndarray``, an ``astropy.units.Quantity``,
        or a ``pint.Quantity``.  Plain numerics are returned as-is (assumed to
        already be in SI).
    expected_physical_type:
        One of ``"mass"``, ``"angle"``, or ``"dimensionless"``.  Used only to
        produce informative error messages; the caller is responsible for
        verifying correctness.

    Returns
    -------
    numpy.ndarray
        1-D float64 array in SI units.
    """
    si_unit = _SI_UNIT[expected_physical_type]

    if hasattr(value, "to"):
        # astropy: .to(unit).value  |  pint: .to(unit).magnitude
        converted = value.to(si_unit) if si_unit else value
        raw = getattr(converted, "value", _MISSING)
        if raw is _MISSING:
            raw = getattr(converted, "magnitude", _MISSING)
        if raw is _MISSING:
            raise TypeError(
                f"Cannot extract numeric value from {type(value).__qualname__}. "
                "Expected an astropy.units.Quantity or pint.Quantity."
            )
        return np.atleast_1d(np.asarray(raw, dtype=np.float64))

    return np.atleast_1d(np.asarray(value, dtype=np.float64))


def to_kg(value) -> np.ndarray:
    """Convert *value* (mass) to kilograms."""
    return _to_si(value, "mass")


def to_rad(value) -> np.ndarray:
    """Convert *value* (angle) to radians."""
    return _to_si(value, "angle")


def to_dimensionless(value) -> np.ndarray:
    """Strip units from a dimensionless *value*."""
    return _to_si(value, "dimensionless")


def to_hz(value) -> np.ndarray:
    """Convert *value* (frequency) to hertz."""
    return _to_si(value, "frequency")
