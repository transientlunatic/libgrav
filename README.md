libgrav
======

> Fair fa' your honest, sonsie face

libgrav is a collection of mathematical and physical routines for use in gravitational-wave astronomy and astrophysics.
It is written in Rust and is designed for accuracy, speed, and accessibility.

libgrav doesn't do data analysis, but you can use it as the core of a more complex algorithm.
It's intended to provide the building blocks which keep being redeveloped and reimplemented elsewhere.
It's the spiritual successor to the LALSuite library, but with a much more modern design and implementation.

What's in libgrav?
------------------

- Parameter transforms and conversions (binary system: total mass, mass ratio, symmetric mass ratio, chirp mass, χ_eff, χ_p)
- Useful units and constants (solar mass, SI throughout)
- Compile-time physical unit safety via [`uom`](https://crates.io/crates/uom)

Language interfaces
-------------------

| Language       | Status      | Notes |
|---------------|-------------|-------|
| Rust           | ✅ stable   | core crate `grav` |
| Python         | ✅ stable   | `pip install libgrav`; numpy, astropy, pint, JAX all supported |
| JavaScript/TypeScript | ✅ stable | `npm install grav-wasm`; vectorised `Float64Array` API + scalar helpers |
| Julia          | ✅ stable   | `ccall` into `libgrav`; broadcasting works natively |
| C / C++        | ✅ stable   | link `libgrav`, include `bindings/julia/include/grav.h` |
| Fortran        | ✅ stable   | `iso_c_binding` + `bind(C)` — no shim needed |
| Go             | ✅ stable   | `cgo` with `#cgo LDFLAGS` pointing at `libgrav` |
| R              | ✅ stable   | `R CMD INSTALL bindings/r`; vectorised, testthat suite |
| MATLAB         | ✅ stable   | `loadlibrary` / `calllib` using the C header; no compilation needed |

Installation
------------

### Python

```bash
pip install libgrav
```

Works with plain numpy arrays, astropy quantities, pint quantities, and JAX arrays.
JAX integration uses `jax.pure_callback` so Rust functions are callable inside JIT-compiled code.

```python
import numpy as np
import libgrav
import libgrav.binary   # domain-organised submodule

MSUN = 1.988_416e30  # kg

m1 = np.array([30.0]) * MSUN
m2 = np.array([30.0]) * MSUN

# Recommended: access via the binary submodule
print(libgrav.binary.chirp_mass(m1, m2) / MSUN)   # ~26.1 M☉

# Top-level shortcut (backward compatible)
print(libgrav.chirp_mass(m1, m2) / MSUN)
```

### JavaScript / TypeScript

```bash
npm install grav-wasm
```

```ts
// Domain-organised import (recommended)
import * as binary from 'grav-wasm/binary';

const mc = binary.chirp_mass_scalar(30 * binary.MSUN, 30 * binary.MSUN);   // kg
```

```ts
// Top-level import also works (backward compatible)
import { MSUN, chirp_mass_scalar, chirp_mass } from 'grav-wasm/libgrav';

const mc_arr = chirp_mass(new Float64Array([30 * MSUN]), new Float64Array([30 * MSUN]));
```

### Julia

```bash
# Build the shared library
cargo build --release -p grav-capi
```

```julia
import Pkg
Pkg.develop(path="bindings/julia")

using Grav

# Recommended: access via the Binary submodule
mc = Grav.Binary.chirp_mass(30.0 * MSUN, 30.0 * MSUN)   # scalar
mc = Grav.Binary.chirp_mass.(m1_array, m2_array)          # vectorised via broadcasting

# Top-level shortcut (backward compatible)
mc = chirp_mass(30.0 * MSUN, 30.0 * MSUN)
```

### Rust

```toml
[dependencies]
grav = "0.1"
```

```rust
use grav::binary;
use uom::si::f64::Mass;
use uom::si::mass::kilogram;

let m = Mass::new::<kilogram>(30.0 * grav::binary::MSUN);
let mc = binary::chirp_mass(m, m);
```

Development
-----------

```bash
# Run all core tests
cargo test -p grav

# Build Python bindings (requires maturin)
pip install maturin
maturin develop --extras dev --manifest-path bindings/python/pyproject.toml

# Build WASM package (requires wasm-pack)
wasm-pack build --target bundler bindings/wasm
```


## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
