"""
    Grav

Julia interface to the Grav Rust library for gravitational-wave binary
parameter computations.

All functions accept and return SI values (`Float64`, kilograms for masses,
radians for angles, dimensionless otherwise).

Functions are organised by physics domain into submodules:

- [`Grav.Binary`](@ref) — compact binary parameter conversions (masses, spins)

All functions are also exported from the top-level `Grav` module for
convenience, so `using Grav; chirp_mass(...)` works without the submodule.

Use Julia broadcasting to apply scalar functions over arrays:

```julia
using Grav

m1 = fill(30.0 * MSUN, 1000)
m2 = fill(30.0 * MSUN, 1000)

mc  = Binary.chirp_mass.(m1, m2)   # via submodule
mc  = chirp_mass.(m1, m2)          # top-level shortcut
eta = symmetric_mass_ratio.(m1, m2)
```
"""
module Grav

# ── Binary submodule ──────────────────────────────────────────────────────────

"""
    Binary

Compact binary system parameter functions for gravitational-wave astronomy.

All functions accept scalar `Float64` values in SI units (kg for mass, radians
for angles).  Vectorisation is handled by Julia broadcasting:

```julia
using Grav

mc = Binary.chirp_mass.(m1_array, m2_array)
```
"""
module Binary

export MSUN,
       total_mass, mass_ratio, symmetric_mass_ratio, chirp_mass,
       masses_from_chirp_mass_q, masses_from_chirp_mass_eta,
       chi_eff, chi_p,
       spin_components, orbital_angular_momentum, transform_precessing_spins

# ── shared library location ───────────────────────────────────────────────────
#
# Development layout (monorepo):
#   cargo build --release -p grav-capi
#   → <repo-root>/target/release/libgrav.{so,dylib,dll}
#
# For a registered Julia package the library should instead be supplied by a
# companion JLL package created with BinaryBuilder.jl.

const _REPO_ROOT = joinpath(@__DIR__, "..", "..", "..")
const _LIB = joinpath(_REPO_ROOT, "target", "release", "libgrav")

# ── constants ─────────────────────────────────────────────────────────────────

"Solar mass in kilograms."
const MSUN::Float64 = 1.988_416e30

# ── binary parameters ─────────────────────────────────────────────────────────

"""
    total_mass(m1_kg, m2_kg) -> Float64

Total mass ``M = m_1 + m_2`` in kilograms.
"""
function total_mass(m1_kg::Float64, m2_kg::Float64)::Float64
    ccall((:grav_total_mass, _LIB), Float64, (Float64, Float64), m1_kg, m2_kg)
end

"""
    mass_ratio(m1_kg, m2_kg) -> Float64

Mass ratio ``q = m_2 / m_1`` (dimensionless, requires ``m_1 \\geq m_2``).
"""
function mass_ratio(m1_kg::Float64, m2_kg::Float64)::Float64
    ccall((:grav_mass_ratio, _LIB), Float64, (Float64, Float64), m1_kg, m2_kg)
end

"""
    symmetric_mass_ratio(m1_kg, m2_kg) -> Float64

Symmetric mass ratio ``\\eta = m_1 m_2 / M^2 \\in (0, 1/4]`` (dimensionless).
"""
function symmetric_mass_ratio(m1_kg::Float64, m2_kg::Float64)::Float64
    ccall((:grav_symmetric_mass_ratio, _LIB), Float64, (Float64, Float64), m1_kg, m2_kg)
end

"""
    chirp_mass(m1_kg, m2_kg) -> Float64

Chirp mass ``\\mathcal{M} = (m_1 m_2)^{3/5} / M^{1/5}`` in kilograms.
"""
function chirp_mass(m1_kg::Float64, m2_kg::Float64)::Float64
    ccall((:grav_chirp_mass, _LIB), Float64, (Float64, Float64), m1_kg, m2_kg)
end

"""
    masses_from_chirp_mass_q(mc_kg, q) -> Tuple{Float64, Float64}

Component masses ``(m_1, m_2)`` in kilograms from chirp mass ``\\mathcal{M}``
(kg) and mass ratio ``q = m_2/m_1 \\in (0, 1]``.

Returns `(m1_kg, m2_kg)` with `m1 \u2265 m2`.
"""
function masses_from_chirp_mass_q(mc_kg::Float64, q::Float64)::Tuple{Float64,Float64}
    m1 = ccall((:grav_m1_from_mc_q, _LIB), Float64, (Float64, Float64), mc_kg, q)
    m2 = ccall((:grav_m2_from_mc_q, _LIB), Float64, (Float64, Float64), mc_kg, q)
    (m1, m2)
end

"""
    masses_from_chirp_mass_eta(mc_kg, eta) -> Tuple{Float64, Float64}

Component masses ``(m_1, m_2)`` in kilograms from chirp mass ``\\mathcal{M}``
(kg) and symmetric mass ratio ``\\eta \\in (0, 0.25]``.

Returns `(m1_kg, m2_kg)` with `m1 \u2265 m2`.
"""
function masses_from_chirp_mass_eta(mc_kg::Float64, eta::Float64)::Tuple{Float64,Float64}
    m1 = ccall((:grav_m1_from_mc_eta, _LIB), Float64, (Float64, Float64), mc_kg, eta)
    m2 = ccall((:grav_m2_from_mc_eta, _LIB), Float64, (Float64, Float64), mc_kg, eta)
    (m1, m2)
end

"""
    chi_eff(m1_kg, m2_kg, a1, a2, tilt1, tilt2) -> Float64

Effective inspiral spin ``\\chi_\\mathrm{eff} \\in [-1, 1]``.

- `a1`, `a2`: dimensionless spin magnitudes (0–1)
- `tilt1`, `tilt2`: spin tilt angles in radians (0–π)
"""
function chi_eff(
    m1_kg::Float64, m2_kg::Float64,
    a1::Float64, a2::Float64,
    tilt1::Float64, tilt2::Float64,
)::Float64
    ccall(
        (:grav_chi_eff, _LIB), Float64,
        (Float64, Float64, Float64, Float64, Float64, Float64),
        m1_kg, m2_kg, a1, a2, tilt1, tilt2,
    )
end

"""
    chi_p(m1_kg, m2_kg, a1, a2, tilt1, tilt2) -> Float64

Effective precession spin ``\\chi_p \\in [0, 1]``.

Requires ``m_1 \\geq m_2``.

- `a1`, `a2`: dimensionless spin magnitudes (0–1)
- `tilt1`, `tilt2`: spin tilt angles in radians (0–π)
"""
function chi_p(
    m1_kg::Float64, m2_kg::Float64,
    a1::Float64, a2::Float64,
    tilt1::Float64, tilt2::Float64,
)::Float64
    ccall(
        (:grav_chi_p, _LIB), Float64,
        (Float64, Float64, Float64, Float64, Float64, Float64),
        m1_kg, m2_kg, a1, a2, tilt1, tilt2,
    )
end

# ── spin components ──────────────────────────────────────────────────────────

"""
    spin_components(a1, a2, tilt1, tilt2, phi12) -> NTuple{6,Float64}

Cartesian spin components in the L-frame:

```
S1 = a1 .* (sin(tilt1), 0, cos(tilt1))
S2 = a2 .* (sin(tilt2)*cos(phi12), sin(tilt2)*sin(phi12), cos(tilt2))
```

- `a1`, `a2`: dimensionless spin magnitudes (0–1)
- `tilt1`, `tilt2`: spin tilt angles from L_N in radians (0–π)
- `phi12`: azimuthal angle of spin 2 relative to spin 1 in radians

Returns `(S1x, S1y, S1z, S2x, S2y, S2z)`.
"""
function spin_components(
    a1::Float64, a2::Float64,
    tilt1::Float64, tilt2::Float64, phi12::Float64,
)::NTuple{6,Float64}
    (
        ccall((:grav_spin_components_s1x, _LIB), Float64,
              (Float64, Float64, Float64, Float64, Float64), a1, a2, tilt1, tilt2, phi12),
        ccall((:grav_spin_components_s1y, _LIB), Float64,
              (Float64, Float64, Float64, Float64, Float64), a1, a2, tilt1, tilt2, phi12),
        ccall((:grav_spin_components_s1z, _LIB), Float64,
              (Float64, Float64, Float64, Float64, Float64), a1, a2, tilt1, tilt2, phi12),
        ccall((:grav_spin_components_s2x, _LIB), Float64,
              (Float64, Float64, Float64, Float64, Float64), a1, a2, tilt1, tilt2, phi12),
        ccall((:grav_spin_components_s2y, _LIB), Float64,
              (Float64, Float64, Float64, Float64, Float64), a1, a2, tilt1, tilt2, phi12),
        ccall((:grav_spin_components_s2z, _LIB), Float64,
              (Float64, Float64, Float64, Float64, Float64), a1, a2, tilt1, tilt2, phi12),
    )
end

# ── orbital angular momentum ─────────────────────────────────────────────────

"""
    orbital_angular_momentum(m1_kg, m2_kg, f_ref) -> Float64

Newtonian orbital angular momentum magnitude ``|L_N|`` (kg m² s⁻¹) at
reference GW frequency `f_ref` (Hz).
"""
function orbital_angular_momentum(m1_kg::Float64, m2_kg::Float64, f_ref::Float64)::Float64
    ccall(
        (:grav_orbital_angular_momentum, _LIB), Float64,
        (Float64, Float64, Float64),
        m1_kg, m2_kg, f_ref,
    )
end

# ── precessing spin frame transform ──────────────────────────────────────────

"""
    transform_precessing_spins(theta_jn, phi_jl, tilt1, tilt2, phi12,
                                a1, a2, m1_kg, m2_kg, f_ref, phase) -> NTuple{7,Float64}

Transform bilby/LALInference precessing-spin parameters into the inclination
and Cartesian spin components used by waveform generators.  A from-scratch
reimplementation of LALSimulation's
`SimInspiralTransformPrecessingNewInitialConditions` (no LALSuite
dependency); matches its output to within floating-point precision.

- `theta_jn`: inclination of J relative to the line of sight (radians)
- `phi_jl`: azimuth of L_N about J (radians)
- `tilt1`, `tilt2`: spin tilt angles from L_N (radians, 0–π)
- `phi12`: azimuthal angle of spin 2 relative to spin 1 (radians)
- `a1`, `a2`: dimensionless spin magnitudes (0–1)
- `m1_kg`, `m2_kg`: component masses (kg)
- `f_ref`: reference GW frequency (Hz), must be nonzero
- `phase`: reference orbital phase (radians)

Returns `(iota, S1x, S1y, S1z, S2x, S2y, S2z)`.
"""
function transform_precessing_spins(
    theta_jn::Float64, phi_jl::Float64,
    tilt1::Float64, tilt2::Float64, phi12::Float64,
    a1::Float64, a2::Float64,
    m1_kg::Float64, m2_kg::Float64,
    f_ref::Float64, phase::Float64,
)::NTuple{7,Float64}
    (
        ccall((:grav_transform_precessing_spins_iota, _LIB), Float64,
              (Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64),
              theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2, m1_kg, m2_kg, f_ref, phase),
        ccall((:grav_transform_precessing_spins_s1x, _LIB), Float64,
              (Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64),
              theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2, m1_kg, m2_kg, f_ref, phase),
        ccall((:grav_transform_precessing_spins_s1y, _LIB), Float64,
              (Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64),
              theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2, m1_kg, m2_kg, f_ref, phase),
        ccall((:grav_transform_precessing_spins_s1z, _LIB), Float64,
              (Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64),
              theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2, m1_kg, m2_kg, f_ref, phase),
        ccall((:grav_transform_precessing_spins_s2x, _LIB), Float64,
              (Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64),
              theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2, m1_kg, m2_kg, f_ref, phase),
        ccall((:grav_transform_precessing_spins_s2y, _LIB), Float64,
              (Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64),
              theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2, m1_kg, m2_kg, f_ref, phase),
        ccall((:grav_transform_precessing_spins_s2z, _LIB), Float64,
              (Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64),
              theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2, m1_kg, m2_kg, f_ref, phase),
    )
end

end # module Binary

# ── top-level re-exports (convenience shortcuts) ──────────────────────────────

export Binary

export MSUN,
       total_mass, mass_ratio, symmetric_mass_ratio, chirp_mass,
       masses_from_chirp_mass_q, masses_from_chirp_mass_eta,
       chi_eff, chi_p,
       spin_components, orbital_angular_momentum, transform_precessing_spins

const MSUN = Binary.MSUN
const total_mass = Binary.total_mass
const mass_ratio = Binary.mass_ratio
const symmetric_mass_ratio = Binary.symmetric_mass_ratio
const chirp_mass = Binary.chirp_mass
const masses_from_chirp_mass_q = Binary.masses_from_chirp_mass_q
const masses_from_chirp_mass_eta = Binary.masses_from_chirp_mass_eta
const chi_eff = Binary.chi_eff
const chi_p = Binary.chi_p
const spin_components = Binary.spin_components
const orbital_angular_momentum = Binary.orbital_angular_momentum
const transform_precessing_spins = Binary.transform_precessing_spins

end # module Grav
