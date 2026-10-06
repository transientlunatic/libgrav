# Contributing to libgrav

libgrav is a small, focused library. Contributions that add new mathematical
functions, fix bugs, or improve documentation are very welcome.

## Ground rules

- Every new function must be **physically well-defined and dimensionally
  correct**. libgrav uses `uom` for compile-time unit safety — lean into it.
- **No waveforms, no detectors.** libgrav only contains parameter
  transformations and derived quantities. If in doubt, ask first.
- All public API changes require a doc-test, unit tests, and property tests
  (see below).
- Run **all tests** before submitting a pull request.
- Keep commits small and the description clear. Update `CHANGELOG.md`.

---

## Development setup

```bash
git clone https://github.com/transientlunatic/puddin
cd puddin

# Rust toolchain (stable)
rustup show          # verify stable is active

# Core library + CLI
cargo test -p grav
cargo build -p grav-cli

# Python bindings (separate environment)
cd bindings/python
pip install maturin
maturin develop --extras dev
pytest tests/ -v
```

---

## How to add a new function

Adding a function end-to-end touches **six layers**.  Work through them in
order.

### 1. Rust core — `crates/grav/src/binary.rs`

Add the function in the appropriate section (or create a new module if it
belongs to a different physical domain).

**Template:**

```rust
/// One-line summary ($\LaTeX$ formula if short enough).
///
/// Longer explanation if needed.  Cite the defining paper with a bare URL
/// or `[Author YYYY](https://doi.org/...)`.
///
/// # Arguments
///
/// * `m1`, `m2` — component masses ($m_1 \geq m_2$, SI kg via `uom`).
/// * `param`    — description, valid range, units.
///
/// # Examples
///
/// ```
/// use uom::si::f64::Mass;
/// use uom::si::mass::kilogram;
/// use grav::binary::my_new_function;
///
/// const MSUN: f64 = 1.988_416e30;
/// let m1 = Mass::new::<kilogram>(30.0 * MSUN);
/// let m2 = Mass::new::<kilogram>(30.0 * MSUN);
/// let result = my_new_function(m1, m2);
/// assert!((result - EXPECTED).abs() < 1e-10);
/// ```
pub fn my_new_function(m1: Mass, m2: Mass) -> f64 {
    // implementation
}
```

**Rules:**
- Masses in and out must use `uom::si::f64::Mass` (not raw `f64`).
  Dimensionless outputs (ratios, spin parameters) are plain `f64`.
- Use `const MSUN: f64 = 1.988_416e30` (IAU 2015). Do not introduce
  `solar_mass` — it does not exist in `uom` v0.36.
- If `m1 >= m2` is a precondition, add a `debug_assert!` with a clear message.
- Export from `crates/grav/src/lib.rs` if it is a new module; `binary.rs`
  exports are already public via `pub mod binary`.

### 2. Tests — bottom of `binary.rs`

Add at minimum:

```rust
#[test]
fn my_new_function_known_value() {
    // hard-coded known result for a simple case
}

#[test]
fn my_new_function_symmetry() {
    // any physical symmetry that should hold
}

proptest! {
    #[test]
    fn my_new_function_range(
        m1 in 1.0f64..200.0,
        m2 in 1.0f64..200.0,
    ) {
        prop_assume!(m1 >= m2);
        let result = my_new_function(solar(m1), solar(m2));
        // assert invariants (e.g. result is in [0, 1])
        prop_assert!(result >= 0.0 && result <= 1.0);
    }
}
```

Run with `cargo test -p grav`.  All 26+ tests must pass.

### 3. C ABI — `bindings/julia/src/lib.rs`

Add a `#[no_mangle] pub extern "C"` wrapper.  Always scalar, always SI:

```rust
/// My new quantity (dimensionless).
#[no_mangle]
pub extern "C" fn grav_my_new_function(m1_kg: f64, m2_kg: f64) -> f64 {
    binary::my_new_function(kg(m1_kg), kg(m2_kg))
}
```

Update the C header `bindings/julia/include/grav.h`:

```c
/** My new quantity (dimensionless). */
double grav_my_new_function(double m1_kg, double m2_kg);
```

Build and verify: `cargo build --release -p grav-capi`.

### 4. CLI — `crates/grav-cli/src/main.rs`

Add a variant to the `Command` enum and a match arm in `main()`:

```rust
/// My new quantity — one-line description
#[command(alias = "mnf")]
MyNewFunction {
    /// Primary mass (heavier component, m1 ≥ m2)
    m1: f64,
    /// Secondary mass
    m2: f64,
},
```

```rust
Command::MyNewFunction { m1, m2 } => {
    let r = my_new_function(mass(m1, si), mass(m2, si));
    (r, "my_new_function", "")   // ("", "") for dimensionless
}
```

If the output carries mass units, pattern-match `si` and set the unit string
to `"Msun"` or `"kg"` as appropriate (see `ChirpMass` for the pattern).

Test: `cargo build -p grav-cli && ./target/debug/grav my-new-function 30 30`.

### 5. Python — `bindings/python/src/lib.rs` and `tests/`

Add a `#[pyfunction]` following the array-in / array-out pattern:

```rust
/// My new quantity (dimensionless).
#[pyfunction]
fn my_new_function<'py>(
    py: Python<'py>,
    m1: PyReadonlyArray1<'py, f64>,
    m2: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let masses1 = array_to_masses(&m1);
    let masses2 = array_to_masses(&m2);
    let result: Vec<f64> = masses1
        .into_iter()
        .zip(masses2)
        .map(|(a, b)| binary::my_new_function(a, b))
        .collect();
    Ok(result.into_pyarray_bound(py))
}
```

Register it in `_libgrav`:

```rust
m.add_function(wrap_pyfunction!(my_new_function, m)?)?;
```

Add a test in `bindings/python/tests/test_binary.py`:

```python
def test_my_new_function_known_value():
    m = np.array([30.0 * MSUN])
    result = libgrav.my_new_function(m, m)
    assert abs(result[0] - EXPECTED) < 1e-10
```

Build and test:

```bash
cd bindings/python
maturin develop --extras dev
pytest tests/ -v
```

### 6. Julia — `bindings/julia/src/Grav.jl` and `test/runtests.jl`

Add a `ccall` wrapper:

```julia
"""
    my_new_function(m1_kg, m2_kg) -> Float64

One-line description.
"""
function my_new_function(m1_kg::Float64, m2_kg::Float64)::Float64
    ccall((:grav_my_new_function, _LIB), Float64, (Float64, Float64), m1_kg, m2_kg)
end
```

Export it at the top of the module:

```julia
export MSUN,
       total_mass, mass_ratio, symmetric_mass_ratio, chirp_mass,
       chi_eff, chi_p, my_new_function
```

Add a test case in `test/runtests.jl`:

```julia
@testset "my_new_function" begin
    @test isapprox(my_new_function(30MSUN, 30MSUN), EXPECTED, rtol=1e-10)
end
```

Run: `julia --project=bindings/julia bindings/julia/test/runtests.jl`.

### 7. R (if applicable) — `bindings/r/src/grav_r.c` and `R/grav.R`

Add a `.C()`-compatible shim in `grav_r.c`:

```c
void r_grav_my_new_function(double *m1_kg, double *m2_kg, double *result) {
    *result = grav_my_new_function(*m1_kg, *m2_kg);
}
```

Register it in the `cMethods` table:

```c
{"r_grav_my_new_function", (DL_FUNC) &r_grav_my_new_function, 3},
```

Add the public API in `R/grav.R`:

```r
#' My new quantity
#'
#' @param m1_kg Primary mass (kg)
#' @param m2_kg Secondary mass (kg)
#' @return Result (dimensionless numeric vector)
#' @export
my_new_function <- Vectorize(function(m1_kg, m2_kg) {
  .C("r_grav_my_new_function",
     m1_kg  = as.double(m1_kg),
     m2_kg  = as.double(m2_kg),
     result = double(1L),
     PACKAGE = "libgrav")$result
})
```

Export the symbol in `NAMESPACE`:

```r
export(my_new_function)
```

Test:

```bash
cargo build --release -p grav-capi
GRAV_LIB=$(pwd)/target/release R CMD INSTALL bindings/r
Rscript bindings/r/tests/testthat.R
```

---

## Docs

### Theory page

If your function needs mathematical background, add or extend a page under
`docs/theory/`:

```bash
docs/theory/binary_parameters.md    # existing: mass & spin parameters
docs/theory/cosmology.md            # new, if adding cosmological functions
```

Use KaTeX-compatible `$$...$$` blocks for display equations and `$...$` for
inline math.

### Interface docs

`docs/interfaces.md` documents usage in every language. If any interface
has non-obvious behaviour for your function (e.g. a different argument
order, a broadcasting note), add a short note in the relevant section.

Build the docs locally to check for broken references:

```bash
pip install -r docs/requirements.txt
sphinx-build -n -b html docs/ docs/_build/html
```

---

## Checklist before opening a pull request

- [ ] `cargo test -p grav` — all tests pass
- [ ] `cargo clippy -p grav -p grav-capi -p grav-cli -- -D warnings` — zero warnings
- [ ] `cargo fmt --all --check` — no formatting changes needed
- [ ] Python tests pass (`cd bindings/python && maturin develop && pytest`)
- [ ] Julia tests pass
- [ ] R tests pass (if R binding was updated)
- [ ] `sphinx-build -n -b html docs/ docs/_build/html` — no warnings
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
