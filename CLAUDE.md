# Puddin — Claude context

Small, focused Rust library of mathematical primitives for gravitational-wave
astronomy.  No waveforms, no detectors.  Currently implements binary system
parameter conversions (`chirp mass`, `mass ratio`, `symmetric mass ratio`,
`chi_eff`, `chi_p`).

## Repository layout

```
crates/puddin/          Core Rust library (uom SI units, proptest)
bindings/julia/         cdylib C ABI — used by Julia, C, C++, Fortran, Go, MATLAB, R
bindings/python/        PyO3/maturin extension — excluded from Cargo workspace
bindings/wasm/          wasm-bindgen — workspace member but requires wasm32 target
bindings/r/             R package wrapping the C ABI via .C() shims; src/rust is a
                        vendored copy of the Rust sources (generated, statically linked)
examples/{c,cpp,fortran,go,matlab}/
docs/                   Sphinx + sphinx-rust + furo + myst-parser + sphinxcontrib-katex
.github/workflows/      ci.yml, release.yml, pages.yml
```

## Essential commands

### Rust
```bash
cargo test -p puddin                          # run core tests (26 tests)
cargo clippy -p puddin -p puddin-julia -p puddin-cli -- -D warnings   # lint workspace crates
cargo fmt --all --check                       # check formatting
cargo doc -p puddin --no-deps                 # generate Rust API docs
```

### CLI (crates/puddin-cli)
```bash
cargo build --release -p puddin-cli           # produces target/release/puddin
cargo install --path crates/puddin-cli        # install to ~/.cargo/bin/puddin

puddin chirp-mass 30 30                       # chirp mass in Msun (stdout: bare number)
puddin chirp-mass 30 30 --verbose             # prints: chirp_mass = 26.1165 Msun
puddin chi-eff 30 30 0.5 0.5 0.0 0.0         # χ_eff
puddin chi-p   30 15 0.8 0.3 0.4 1.2         # χ_p
puddin --si chirp-mass 5.97e30 5.97e30        # masses in kg
```

### Python bindings (must be run from bindings/python — uses maturin)
```bash
cd bindings/python
maturin develop --extras dev
pytest tests/ -v
```

### WASM (requires wasm32-unknown-unknown target)
```bash
cd bindings/wasm
wasm-pack build --target bundler --out-dir pkg
```

### Julia
```bash
cargo build --release -p puddin-julia         # build the shared library first
cd bindings/julia
julia --project=. test/runtests.jl
```

### R
```bash
R CMD INSTALL bindings/r                      # compiles the vendored Rust in src/rust (needs cargo)
cd bindings/r && Rscript tests/testthat.R
python3 scripts/vendor_r.py                   # re-vendor after changing crates/puddin or bindings/julia
python3 scripts/vendor_r.py --check           # CI check that src/rust is in sync
# Optional: link a prebuilt shared lib instead of building Rust
cargo build --release -p puddin-julia && PUDDIN_LIB=$(pwd)/target/release R CMD INSTALL bindings/r
```

### Releasing
```bash
git tag v0.2.0 && git push origin v0.2.0   # that's it
python3 scripts/set_version.py v0.2.0      # optional: stamp versions locally
```
The tag is the source of truth; `release.yml` stamps it into `Cargo.toml`,
`bindings/python/{Cargo,pyproject}.toml`, `Project.toml` and the R `DESCRIPTION`
in every job.  Versions committed in the repo are placeholders.

### Docs
```bash
pip install -r docs/requirements.txt
sphinx-build -n -b html docs/ docs/_build/html
```

## Key constraints

- **`bindings/python` is excluded from the Cargo workspace** — it must be built
  through `maturin`.  Never add it to `[workspace.members]`.  Use `-p puddin`
  not `--workspace` in Rust CI commands.
- **`uom` v0.36 has no `solar_mass` unit** — use `kilogram` +
  `pub const MSUN: f64 = 1.988_416e30`.
- **`chi_p` convention**: `m1 >= m2` is required.  Enforced with
  `debug_assert!` in release builds and `prop_assume!` in property tests.
- **`opt-level = "z"` only applies to `puddin-wasm`** via
  `[profile.release.package.puddin-wasm]`.  The global `[profile.release]`
  only sets `lto = true`.
- **WASM clippy** runs under the wasm-build CI job (full `wasm-pack build`),
  not under `rust-checks`.  The `rust-checks` clippy step covers
  `-p puddin -p puddin-julia` only.
- **C ABI shared library**: `bindings/julia` produces
  `libpuddin_julia.{so,dylib,dll}` and `bindings/julia/include/puddin.h`.
  All non-Rust/non-Python consumers (C, C++, Fortran, Go, MATLAB, R) depend
  on this library.

## Conventions

- SI units everywhere in the Rust API — masses in kg, angles in radians.
- All public functions have doc-tests with a GW150914-like 30+30 M☉ example.
- Examples in `examples/` all use the same 30+30 M☉ binary for
  cross-language comparison.
- Dual-licensed MIT OR Apache-2.0.
- Test-driven development with `proptest` for the Rust core library; `testthat`
  for R; `pytest` for Python; and native test scripts for Julia, C, C++,
  Fortran, Go, and MATLAB/Octave.
- Documentation built with Sphinx + sphinx-kentigern + myst-parser + sphinxcontrib-katex + sphinx-rust.
- Frequent commits with clear messages, and a detailed `CHANGELOG.md` for release notes. Before making a commit all tests must pass, and changes must not cause tests to fail. 
