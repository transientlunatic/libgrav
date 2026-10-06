/**
 * Binary system parameter functions for the libgrav WASM package.
 *
 * This module re-exports all compact binary parameter functions from
 * `grav-wasm` so that ECMAScript consumers can use a structured import:
 *
 * ```ts
 * import * as binary from 'grav-wasm/binary';
 *
 * const mc = binary.chirp_mass_scalar(30 * binary.MSUN, 30 * binary.MSUN);
 * ```
 *
 * All symbols exported here are identical to those in the top-level
 * `grav-wasm/libgrav` module.  The `binary` module exists purely to give
 * users a physics-domain-organised entry point that mirrors the Python
 * `libgrav.binary` submodule.
 */

export {
  MSUN,
  total_mass,
  total_mass_scalar,
  mass_ratio,
  mass_ratio_scalar,
  symmetric_mass_ratio,
  symmetric_mass_ratio_scalar,
  chirp_mass,
  chirp_mass_scalar,
  masses_from_chirp_mass_q,
  masses_from_chirp_mass_q_scalar,
  masses_from_chirp_mass_eta,
  masses_from_chirp_mass_eta_scalar,
  chi_eff,
  chi_eff_scalar,
  chi_p,
  chi_p_scalar,
} from "./libgrav.js";
