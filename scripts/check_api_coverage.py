#!/usr/bin/env python3
"""
check_api_coverage.py — Verify that every public libgrav function listed in
api.toml is implemented in every binding layer.

Exit code 0 = all present.  Non-zero = at least one gap found.

Requires Python 3.11+ (tomllib is in the stdlib).
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# All layers that must be declared for every function entry.
# A missing key in api.toml is treated as a failure, not a silent skip.
REQUIRED_LAYERS = ("cli", "python", "julia", "r", "wasm")


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def ensure(failures: list[str], label: str, rel_path: str, needle: str) -> None:
    """Append a failure message if *needle* is not found in the file."""
    try:
        content = read(rel_path)
    except FileNotFoundError:
        failures.append(f"  FILE NOT FOUND  [{label}]  {rel_path}")
        return
    if needle not in content:
        failures.append(f"  MISSING  [{label}]  '{needle}'  in  {rel_path}")


def as_list(value: object) -> list[str]:
    """Return *value* as a list regardless of whether it is a str or list."""
    return [value] if isinstance(value, str) else list(value)


def main() -> int:
    manifest_path = ROOT / "api.toml"
    with manifest_path.open("rb") as fh:
        manifest = tomllib.load(fh)

    failures: list[str] = []

    for fn in manifest["function"]:
        name: str = fn["rust_name"]
        print(f"Checking: {name}")

        # ── Rust core ────────────────────────────────────────────────────────
        ensure(failures, "rust",
               "crates/grav/src/binary.rs",
               f"pub fn {name}(")

        # ── C ABI  (library source + header) ────────────────────────────────
        for sym in as_list(fn.get("c_abi", [])):
            ensure(failures, "c_abi",
                   "bindings/julia/src/lib.rs",
                   f"{sym}(")
            ensure(failures, "c_header",
                   "bindings/julia/include/grav.h",
                   f"{sym}(")

        # ── Require all binding-layer keys to be declared ────────────────────
        for layer in REQUIRED_LAYERS:
            if layer not in fn:
                failures.append(
                    f"  MISSING KEY  [{layer}]  '{name}'  in  api.toml "
                    f"(add the key or mark it skip_layers = [\"{layer}\"])"
                )

        # ── CLI ──────────────────────────────────────────────────────────────
        if "cli" in fn:
            ensure(failures, "cli",
                   "crates/grav-cli/src/main.rs",
                   fn["cli"])

        # ── Python ───────────────────────────────────────────────────────────
        if "python" in fn:
            # Rust extension must register the function
            ensure(failures, "python_ext",
                   "bindings/python/src/lib.rs",
                   f"wrap_pyfunction!({fn['python']}")
            # Public Python layer must expose it
            ensure(failures, "python_public",
                   "bindings/python/python/libgrav/__init__.py",
                   fn["python"])
            # JAX wrapper must have a matching implementation
            ensure(failures, "python_jax",
                   "bindings/python/python/libgrav/jax_wrapper.py",
                   f"def {fn['python']}(")

        # ── Julia ────────────────────────────────────────────────────────────
        if "julia" in fn:
            ensure(failures, "julia",
                   "bindings/julia/src/Grav.jl",
                   f"function {fn['julia']}(")

        # ── R (R source + NAMESPACE + native C shim) ─────────────────────────
        if "r" in fn:
            ensure(failures, "r_source",
                   "bindings/r/R/grav.R",
                   fn["r"])
            ensure(failures, "r_namespace",
                   "bindings/r/NAMESPACE",
                   fn["r"])
            # Verify the native C shim that adapts the C ABI to R's .C() convention
            for sym in as_list(fn.get("c_abi", [])):
                r_sym = sym.replace("grav_", "r_grav_")
                ensure(failures, "r_c_shim",
                       "bindings/r/src/grav_r.c",
                       f"{r_sym}(")

        # ── WASM (TypeScript wrapper + Rust wasm_bindgen export) ────────────
        if "wasm" in fn:
            ensure(failures, "wasm_ts",
                   "bindings/wasm/js/libgrav.ts",
                   fn["wasm"])
            # Verify the underlying wasm_bindgen Rust source calls the core function
            ensure(failures, "wasm_rust",
                   "bindings/wasm/src/lib.rs",
                   f"binary::{name}")

    print()
    if failures:
        print(f"FAILED — {len(failures)} missing binding(s):\n")
        for msg in failures:
            print(msg)
        return 1

    print(f"OK — all {len(manifest['function'])} function(s) present in every layer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
