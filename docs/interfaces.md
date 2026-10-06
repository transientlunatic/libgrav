# Language interfaces

Puddin's core library is written in Rust and exposed to other languages through
thin binding layers. All interfaces accept and return values in **SI units**
(kilograms for mass, radians for angles). Unit conversion helpers are provided
for the Python interface.

Functions are organised by physics domain. The current domain is:

- **`binary`** — compact binary system parameters (masses, spins)

Each language exposes this as a submodule or namespace so callers can write
`puddin.binary.chirp_mass(...)`, `Puddin.Binary.chirp_mass(...)`,
`puddin::binary::chirp_mass(...)`, etc.  All top-level names are also
re-exported for backward compatibility.

---

## Command line

Install the `puddin` CLI with Cargo:

```bash
cargo install --git https://github.com/transientlunatic/puddin puddin-cli
# or, from a local checkout:
cargo install --path crates/puddin-cli
```

Pre-built binaries for Linux, macOS, and Windows are attached to each
[GitHub Release](https://github.com/transientlunatic/puddin/releases).

### Usage

Masses default to solar masses (M☉); use `--si` to switch to kilograms.
Output is a bare number on stdout — suitable for shell pipelines.
Add `--verbose` (`-v`) for a labelled result with units.

```bash
# Chirp mass of a 30+30 Msun binary
puddin chirp-mass 30 30
# → 26.116516898883617

puddin chirp-mass 30 30 --verbose
# → chirp_mass = 26.116516898883617 Msun

# Spin parameters
puddin chi-eff 30 30 0.5 0.5 0.0 0.0    # both aligned at 0.5 → 0.5
puddin chi-p   30 15 0.8 0.3 0.4 1.2

# Masses in kilograms
puddin --si chirp-mass 5.97e30 5.97e30

# Compose with standard Unix tools
MC=$(puddin chirp-mass 30 30)
echo "Mc = $MC Msun"
```

### Subcommands

| Subcommand       | Aliases       | Arguments              | Output      |
|------------------|---------------|------------------------|-------------|
| `total-mass`     | `m`           | `m1 m2`                | M (Msun/kg) |
| `mass-ratio`     | `q`           | `m1 m2`                | q           |
| `sym-mass-ratio` | `eta`         | `m1 m2`                | η           |
| `chirp-mass`     | `mc`          | `m1 m2`                | ℳ (Msun/kg) |
| `chi-eff`        | `chieff`      | `m1 m2 a1 a2 t1 t2`   | χ_eff       |
| `chi-p`          | `chip`        | `m1 m2 a1 a2 t1 t2`   | χ_p         |

`t1`, `t2` are tilt angles in radians.  `a1`, `a2` are dimensionless spin
magnitudes in [0, 1].  Both spin subcommands require `m1 ≥ m2`.

---

## Python

Install from PyPI:

```bash
pip install puddin
```

Optional extras:

```bash
pip install "puddin[pint]"   # pint unit support
pip install "puddin[jax]"    # JAX / JIT support
```

### numpy arrays (SI)

```python
import numpy as np
import puddin
import puddin.binary   # domain-organised submodule

MSUN = 1.988_416e30  # kg

m1 = np.full(1000, 30.0 * MSUN)
m2 = np.full(1000, 30.0 * MSUN)

# Recommended: use the binary submodule
mc  = puddin.binary.chirp_mass(m1, m2)
eta = puddin.binary.symmetric_mass_ratio(m1, m2)

# Top-level shortcuts still work (backward compatible)
mc  = puddin.chirp_mass(m1, m2)
```

### astropy quantities

```python
import astropy.units as u
import puddin

m1 = 30 * u.Msun
m2 = 30 * u.Msun

mc = puddin.chirp_mass(m1, m2)   # returns astropy Quantity in kg
```

### pint quantities

```python
import pint
ureg = pint.UnitRegistry()

m1 = 30 * ureg.solar_mass
mc = puddin.chirp_mass(m1, m1)
```

### JAX

Puddin functions are usable inside `jax.jit`-compiled code via
`jax.pure_callback`. Analytical `custom_vjp` rules are registered so that
gradients flow through correctly.

```python
import jax
import jax.numpy as jnp
import puddin

MSUN = 1.988_416e30

@jax.jit
def log_chirp_mass(m1, m2):
    return jnp.log(puddin.chirp_mass(m1, m2))

m = jnp.array(30.0 * MSUN)
val, grad = jax.value_and_grad(log_chirp_mass)(m, m)
```

---

## JavaScript / TypeScript

Install from npm:

```bash
npm install puddin-wasm
```

The package is built with [wasm-pack](https://rustwasm.github.io/wasm-pack/) and
ships TypeScript declaration files. It targets modern ES module bundlers
(Vite, esbuild, webpack 5, Rollup).

### Vectorised API (`Float64Array`)

All core functions accept and return `Float64Array` for batch processing.
Import from the `binary` submodule (recommended) or from the top-level module:

```ts
// Domain-organised import (recommended)
import * as binary from 'puddin-wasm/binary';

const m1 = new Float64Array([30 * binary.MSUN, 10 * binary.MSUN]);
const m2 = new Float64Array([30 * binary.MSUN,  5 * binary.MSUN]);

const mc  = binary.chirp_mass(m1, m2);
const eta = binary.symmetric_mass_ratio(m1, m2);
```

```ts
// Top-level import (backward compatible)
import { chirp_mass, symmetric_mass_ratio, MSUN } from 'puddin-wasm/puddin';

const mc  = chirp_mass(m1, m2);
const eta = symmetric_mass_ratio(m1, m2);
```

### Scalar convenience functions

The TypeScript wrapper layer provides `*_scalar()` helpers that accept and
return plain `number`, useful for single-event interactive work:

```ts
import { chirp_mass_scalar, chi_eff_scalar, MSUN } from 'puddin-wasm/puddin';

const mc = chirp_mass_scalar(30 * MSUN, 30 * MSUN);

const xe = chi_eff_scalar(
  30 * MSUN, 30 * MSUN,  // m1, m2
  0.5, 0.3,              // a1, a2
  0.2, 1.1               // tilt1, tilt2 (radians)
);
```

### Available functions

| Function | Description |
|---|---|
| `total_mass` / `total_mass_scalar` | $M = m_1 + m_2$ |
| `mass_ratio` / `mass_ratio_scalar` | $q = m_2/m_1$ |
| `symmetric_mass_ratio` / `symmetric_mass_ratio_scalar` | $\eta = m_1 m_2/M^2$ |
| `chirp_mass` / `chirp_mass_scalar` | $\mathcal{M} = (m_1 m_2)^{3/5}/M^{1/5}$ |
| `chi_eff` / `chi_eff_scalar` | Effective inspiral spin |
| `chi_p` / `chi_p_scalar` | Effective precession spin ($m_1 \geq m_2$ required) |

---

## Julia

The Julia interface uses [`ccall`](https://docs.julialang.org/en/v1/base/c/#ccall)
to call into a Rust shared library (`libpuddin_julia.so` / `.dylib` / `.dll`).
The C API is **scalar only** — Julia's native broadcasting handles arrays
idiomatically without any extra wrapping.

> **Note on distribution:** The recommended approach for a registered Julia
> package with binary dependencies is a companion JLL package created with
> [BinaryBuilder.jl](https://binarybuilder.org/).  The instructions below
> describe the development workflow from source.

### Build the shared library

```bash
cargo build --release -p puddin-julia
# → target/release/libpuddin_julia.{so,dylib,dll}
```

### Add the package

From the Julia REPL, with the monorepo checked out:

```julia
import Pkg
Pkg.develop(path="bindings/julia")
```

### Usage

```julia
using Puddin

const MSUN = Puddin.MSUN   # 1.988_416e30 kg

# Recommended: use the Binary submodule
mc = Puddin.Binary.chirp_mass(30.0 * MSUN, 30.0 * MSUN)   # ≈ 26.1 M☉ in kg

# Top-level shortcuts also work (backward compatible)
mc = chirp_mass(30.0 * MSUN, 30.0 * MSUN)

# Vectorised via Julia broadcasting — no extra work needed
m1 = rand(1000) .* 50.0 .* MSUN
m2 = rand(1000) .* 50.0 .* MSUN

mc_arr  = Puddin.Binary.chirp_mass.(m1, m2)
eta_arr = Puddin.Binary.symmetric_mass_ratio.(m1, m2)

xe_arr  = Puddin.Binary.chi_eff.(m1, m2,
                                 fill(0.5, 1000), fill(0.3, 1000),
                                 fill(0.2, 1000), fill(1.1, 1000))
```

### Available functions

| Function | Description |
|---|---|
| `total_mass(m1, m2)` | $M = m_1 + m_2$ |
| `mass_ratio(m1, m2)` | $q = m_2/m_1$ |
| `symmetric_mass_ratio(m1, m2)` | $\eta = m_1 m_2/M^2$ |
| `chirp_mass(m1, m2)` | $\mathcal{M} = (m_1 m_2)^{3/5}/M^{1/5}$ |
| `chi_eff(m1, m2, a1, a2, tilt1, tilt2)` | Effective inspiral spin |
| `chi_p(m1, m2, a1, a2, tilt1, tilt2)` | Effective precession spin ($m_1 \geq m_2$) |

All masses in kg, all angles in radians

---

## Rust

Add to `Cargo.toml`:

```toml
[dependencies]
puddin = "0.1"
uom    = { version = "0.36", features = ["f64", "si"] }
```

All public functions use [uom](https://crates.io/crates/uom) quantity types
for compile-time dimensional analysis:

```rust
use puddin::binary::{chirp_mass, chi_eff, MSUN};
use uom::si::f64::Mass;
use uom::si::mass::kilogram;

let m = Mass::new::<kilogram>(30.0 * MSUN);
let mc = chirp_mass(m, m);

// Spin parameters (dimensionless f64)
let xe = chi_eff(m, m, 0.5, 0.3, 0.2, 1.1);
```

Attempting to pass a length where a mass is expected is a **compile error** —
no runtime checks needed.

---

## C

The `bindings/julia` crate produces a standard C shared library.  Include
`bindings/julia/include/puddin.h` to get the function declarations and the
`PUDDIN_MSUN` constant.

```bash
# Build the library
cargo build --release -p puddin-julia

# Compile and link your program
gcc example.c \
    -I bindings/julia/include \
    -L target/release -lpuddin_julia \
    -Wl,-rpath,$(pwd)/target/release \
    -o example -lm
```

```c
#include "puddin.h"
#include <stdio.h>

int main(void) {
    double m1 = 30.0 * PUDDIN_MSUN;
    double m2 = 30.0 * PUDDIN_MSUN;
    printf("Chirp mass: %.4f Msun\n", puddin_chirp_mass(m1, m2) / PUDDIN_MSUN);
    return 0;
}
```

See `examples/c/example.c` in the repository for a complete runnable example with a `Makefile`.

---

## C++

Include `puddin/binary.hpp` to use the `puddin::binary::` namespace (recommended),
or fall back to the flat C API via `puddin.h`.

```bash
g++ example.cpp \
    -I bindings/julia/include \
    -L target/release -lpuddin_julia \
    -Wl,-rpath,$(pwd)/target/release \
    -o example
```

```cpp
#include "puddin/binary.hpp"   // provides puddin::binary:: namespace
#include <iostream>

int main() {
    namespace binary = puddin::binary;
    double m1 = 30.0 * binary::MSUN, m2 = 30.0 * binary::MSUN;
    std::cout << "Chirp mass: " << binary::chirp_mass(m1, m2) / binary::MSUN << " Msun\n";
}
```

See `examples/cpp/example.cpp` in the repository for the full example.

---

## Fortran

The `puddin_binary` Fortran module (in `bindings/julia/include/puddin_binary.f90`)
groups all binary parameter interfaces under a single module, mirroring the
domain-organised structure of the other language interfaces.

```bash
gfortran bindings/julia/include/puddin_binary.f90 example.f90 \
    -L target/release -lpuddin_julia \
    -Wl,-rpath,$(pwd)/target/release \
    -o example
```

```fortran
use puddin_binary

real(c_double), parameter :: MSUN = puddin_binary_MSUN
print *, chirp_mass(30*MSUN, 30*MSUN) / MSUN   ! ~26.1
```

See `examples/fortran/example.f90` in the repository for the full example.

---

## Go

Go's `cgo` interface handles the C ABI transparently.  The example below
wraps the C calls in a `binary` struct that mirrors the domain-organised
structure used in the other language interfaces.

```go
// #cgo CFLAGS:  -I../../bindings/julia/include
// #cgo LDFLAGS: -L../../target/release -lpuddin_julia -Wl,-rpath,../../target/release
// #include "puddin.h"
import "C"
import "fmt"

var binary = struct {
    MSUN      float64
    ChirpMass func(float64, float64) float64
    // ... other fields
}{
    MSUN: float64(C.PUDDIN_MSUN),
    ChirpMass: func(m1, m2 float64) float64 {
        return float64(C.puddin_chirp_mass(C.double(m1), C.double(m2)))
    },
}

func main() {
    m := 30.0 * binary.MSUN
    fmt.Printf("Chirp mass: %.4f Msun\n", binary.ChirpMass(m, m)/binary.MSUN)
}
```

```bash
cd examples/go && go run example.go
```

See `examples/go/example.go` in the repository for the full example.
A reusable `binary` package is also available at `bindings/go/binary/`.

---

## R

The R package wraps the C ABI via a thin `.C()`-compatible shim compiled as
part of the package.  All public functions are vectorised with `Vectorize()`.

### Installation

The package is self-contained: it carries a vendored copy of the Rust sources
(`bindings/r/src/rust`) and compiles and statically links them when installed,
so all you need is a Rust toolchain (`cargo`, `rustc`).

```bash
R CMD INSTALL bindings/r            # from a checkout
# or, from a built tarball / r-universe:
R CMD build bindings/r && R CMD INSTALL Puddin_*.tar.gz
```

To link a prebuilt shared library instead (for example from a release archive),
set `PUDDIN_LIB` to the directory containing `libpuddin_julia`:

```bash
PUDDIN_LIB=/opt/puddin/lib R CMD INSTALL bindings/r
```

After changing the Rust sources, refresh the vendored copy with
`python3 scripts/vendor_r.py` (CI fails if it is out of date).

### Usage

```r
library(Puddin)

# Domain-organised access via the binary environment (recommended)
Puddin::binary$chirp_mass(30 * MSUN, 30 * MSUN) / MSUN   # ~26.1

# Top-level functions still work (backward compatible)
chirp_mass(30 * MSUN, 30 * MSUN) / MSUN

# Vectorised — works automatically
m1 <- c(30, 20, 10) * MSUN
m2 <- c(30, 20,  5) * MSUN
Puddin::binary$chirp_mass(m1, m2) / MSUN

# Spin parameters
Puddin::binary$chi_eff(30*MSUN, 30*MSUN, 0.5, 0.5, 0, 0)   # 0.5
```

> **Note on CRAN distribution:** producing a CRAN package with a compiled
> dependency requires either a companion `r2u` / CRAN binary or distributing
> pre-built binaries via a system dependency.  The current package is suitable
> for direct installation from source.

---

## MATLAB

Use MATLAB's `loadlibrary` / `calllib` to load the shared library directly,
then wrap functions in a struct for domain-organised access.

```matlab
loadlibrary('libpuddin_julia', 'bindings/julia/include/puddin.h', 'alias', 'puddin');

MSUN = 1.988416e30;

% Domain-organised binary struct (recommended)
binary.MSUN      = MSUN;
binary.chirp_mass = @(m1,m2) calllib('puddin','puddin_chirp_mass',m1,m2);
% ... (see examples/matlab/example.m for all fields)

mc = binary.chirp_mass(30*MSUN, 30*MSUN) / MSUN;
fprintf('Chirp mass: %.4f Msun\n', mc);

unloadlibrary('puddin');
```

For batch processing, `arrayfun` provides a vectorised wrapper:

```matlab
mc = arrayfun(@(m) binary.chirp_mass(m*MSUN, m*MSUN)/MSUN, 10:10:50);
```

See `examples/matlab/example.m` in the repository for the full example.
