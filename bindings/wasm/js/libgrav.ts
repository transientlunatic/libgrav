/**
 * TypeScript helpers for the Grav WASM package.
 *
 * The raw WASM module (`grav-wasm`) exposes vectorised functions that accept
 * `Float64Array`.  This wrapper adds:
 *
 *   - `MSUN` – solar mass constant (kg)
 *   - Scalar convenience functions (`_scalar` suffix) for interactive use
 *   - Re-exports of the vectorised forms for NumPy-style workflows
 *
 * Usage (ESM / bundler):
 * ```ts
 * import { MSUN, chirp_mass, chirp_mass_scalar } from 'grav-wasm/libgrav';
 *
 * const mc = chirp_mass_scalar(30 * MSUN, 30 * MSUN);  // kg
 * ```
 */

// Adjust the import path to match the wasm-pack --out-name, if overridden.
import {
  chirp_mass as _chirp_mass,
  chi_eff as _chi_eff,
  chi_p as _chi_p,
  mass_ratio as _mass_ratio,
  masses_from_chirp_mass_q_m1 as _mc_q_m1,
  masses_from_chirp_mass_q_m2 as _mc_q_m2,
  masses_from_chirp_mass_eta_m1 as _mc_eta_m1,
  masses_from_chirp_mass_eta_m2 as _mc_eta_m2,
  symmetric_mass_ratio as _symmetric_mass_ratio,
  total_mass as _total_mass,
} from "./grav_wasm.js";

// ── constants ─────────────────────────────────────────────────────────────────

/** Solar mass in kilograms. */
export const MSUN: number = 1.988_416e30;

// ── vectorised re-exports ─────────────────────────────────────────────────────

/** Total mass M = m₁ + m₂ (kg). */
export const total_mass: (m1: Float64Array, m2: Float64Array) => Float64Array =
  _total_mass;

/** Mass ratio q = m₂ / m₁ (dimensionless). */
export const mass_ratio: (
  m1: Float64Array,
  m2: Float64Array
) => Float64Array = _mass_ratio;

/** Symmetric mass ratio η = m₁m₂/M² (dimensionless). */
export const symmetric_mass_ratio: (
  m1: Float64Array,
  m2: Float64Array
) => Float64Array = _symmetric_mass_ratio;

/** Chirp mass 𝓜 = (m₁m₂)^(3/5) / M^(1/5) (kg). */
export const chirp_mass: (m1: Float64Array, m2: Float64Array) => Float64Array =
  _chirp_mass;

/**
 * Component masses (m₁, m₂) from chirp mass 𝓜 and mass ratio q ∈ (0, 1].
 * Returns a tuple `[m1, m2]` of `Float64Array`s in kilograms.
 */
export function masses_from_chirp_mass_q(
  mc: Float64Array,
  q: Float64Array
): [Float64Array, Float64Array] {
  return [_mc_q_m1(mc, q), _mc_q_m2(mc, q)];
}

/**
 * Component masses (m₁, m₂) from chirp mass 𝓜 and symmetric mass ratio η ∈ (0, 0.25].
 * Returns a tuple `[m1, m2]` of `Float64Array`s in kilograms.
 */
export function masses_from_chirp_mass_eta(
  mc: Float64Array,
  eta: Float64Array
): [Float64Array, Float64Array] {
  return [_mc_eta_m1(mc, eta), _mc_eta_m2(mc, eta)];
}

/** Effective inspiral spin χ_eff ∈ [−1, 1]. */
export const chi_eff: (
  m1: Float64Array,
  m2: Float64Array,
  a1: Float64Array,
  a2: Float64Array,
  tilt1: Float64Array,
  tilt2: Float64Array
) => Float64Array = _chi_eff;

/** Effective precession spin χ_p ∈ [0, 1].  Requires m₁ ≥ m₂. */
export const chi_p: (
  m1: Float64Array,
  m2: Float64Array,
  a1: Float64Array,
  a2: Float64Array,
  tilt1: Float64Array,
  tilt2: Float64Array
) => Float64Array = _chi_p;

// ── scalar convenience wrappers ───────────────────────────────────────────────

const f = (v: number): Float64Array => new Float64Array([v]);

/**
 * Scalar variant of {@link total_mass}.
 * @param m1 - Mass 1 (kg)
 * @param m2 - Mass 2 (kg)
 * @returns Total mass (kg)
 */
export function total_mass_scalar(m1: number, m2: number): number {
  return _total_mass(f(m1), f(m2))[0];
}

/**
 * Scalar variant of {@link mass_ratio}.
 * @param m1 - Primary mass (kg, m1 ≥ m2)
 * @param m2 - Secondary mass (kg)
 * @returns Dimensionless mass ratio
 */
export function mass_ratio_scalar(m1: number, m2: number): number {
  return _mass_ratio(f(m1), f(m2))[0];
}

/**
 * Scalar variant of {@link symmetric_mass_ratio}.
 * @param m1 - Mass 1 (kg)
 * @param m2 - Mass 2 (kg)
 * @returns Dimensionless symmetric mass ratio
 */
export function symmetric_mass_ratio_scalar(m1: number, m2: number): number {
  return _symmetric_mass_ratio(f(m1), f(m2))[0];
}

/**
 * Scalar variant of {@link chirp_mass}.
 * @param m1 - Mass 1 (kg)
 * @param m2 - Mass 2 (kg)
 * @returns Chirp mass (kg)
 */
export function chirp_mass_scalar(m1: number, m2: number): number {
  return _chirp_mass(f(m1), f(m2))[0];
}

/**
 * Scalar variant of {@link masses_from_chirp_mass_q}.
 * @param mc - Chirp mass (kg)
 * @param q  - Mass ratio q ∈ (0, 1]
 * @returns Tuple [m1, m2] in kilograms
 */
export function masses_from_chirp_mass_q_scalar(
  mc: number,
  q: number
): [number, number] {
  return [_mc_q_m1(f(mc), f(q))[0], _mc_q_m2(f(mc), f(q))[0]];
}

/**
 * Scalar variant of {@link masses_from_chirp_mass_eta}.
 * @param mc  - Chirp mass (kg)
 * @param eta - Symmetric mass ratio η ∈ (0, 0.25]
 * @returns Tuple [m1, m2] in kilograms
 */
export function masses_from_chirp_mass_eta_scalar(
  mc: number,
  eta: number
): [number, number] {
  return [_mc_eta_m1(f(mc), f(eta))[0], _mc_eta_m2(f(mc), f(eta))[0]];
}

/**
 * Scalar variant of {@link chi_eff}.
 * @param m1    - Mass 1 (kg)
 * @param m2    - Mass 2 (kg)
 * @param a1    - Spin magnitude 1 (0–1)
 * @param a2    - Spin magnitude 2 (0–1)
 * @param tilt1 - Tilt angle 1 (radians)
 * @param tilt2 - Tilt angle 2 (radians)
 * @returns Effective inspiral spin
 */
export function chi_eff_scalar(
  m1: number,
  m2: number,
  a1: number,
  a2: number,
  tilt1: number,
  tilt2: number
): number {
  return _chi_eff(f(m1), f(m2), f(a1), f(a2), f(tilt1), f(tilt2))[0];
}

/**
 * Scalar variant of {@link chi_p}.  Requires m1 ≥ m2.
 * @param m1    - Primary mass (kg, m1 ≥ m2)
 * @param m2    - Secondary mass (kg)
 * @param a1    - Spin magnitude 1 (0–1)
 * @param a2    - Spin magnitude 2 (0–1)
 * @param tilt1 - Tilt angle 1 (radians)
 * @param tilt2 - Tilt angle 2 (radians)
 * @returns Effective precession spin
 */
export function chi_p_scalar(
  m1: number,
  m2: number,
  a1: number,
  a2: number,
  tilt1: number,
  tilt2: number
): number {
  return _chi_p(f(m1), f(m2), f(a1), f(a2), f(tilt1), f(tilt2))[0];
}
