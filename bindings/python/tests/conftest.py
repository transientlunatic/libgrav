"""Shared test fixtures for the libgrav Python binding tests."""

from __future__ import annotations

import numpy as np

MSUN_KG = 1.989e30  # approximate solar mass in kg


def sol(m):
    """Return mass m (solar masses) as a plain SI float array (kg)."""
    return np.atleast_1d(np.float64(m * MSUN_KG))
