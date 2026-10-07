# Changelog

## Unreleased

### Release pipeline fixes (found by the `v0.1.0-rc.1` dry run)
- macOS x86_64 C library and CLI are cross-compiled from `macos-latest`; the
  `macos-13` runner image no longer exists, so those jobs queued forever.
- Python wheels use the stable ABI (`abi3-py310`): one wheel per platform covers
  Python >= 3.10.  This fixes the manylinux build ("couldn't find any python
  interpreters") and means macOS/Windows wheels are no longer tied to the single
  interpreter on the runner.
- Removed the `docker`-in-container QEMU step that broke the aarch64 wheel; it
  is cross-compiled by the manylinux cross image instead.
- The npm package now includes the compiled TypeScript wrappers
  (`grav-wasm/libgrav`, `grav-wasm/binary`) with subpath exports
  (`scripts/finish_wasm_pkg.py`); previously wasm-pack's `files` list omitted them.

### R package
- Now self-contained: the Rust sources are vendored into `bindings/r/src/rust`
  (`scripts/vendor_r.py`) and compiled and statically linked at install time, so
  `R CMD build` tarballs install anywhere with a Rust toolchain (r-universe
  ready).  `GRAV_LIB` still links a prebuilt shared library.
- `grav-capi` also builds as a `staticlib`.

### CI
- R job builds and installs from a source tarball and checks the vendored sources are in sync.
- R job installs testthat from the default (binary) repository and fails loudly if it is missing.

### Release pipeline
- Fixed the PyPI publish action (`pypa/gh-action-pypi-publish`).
- The git tag is the single source of truth for the version: every release job
  stamps it into all manifests with `scripts/set_version.py` (so there is no
  need to bump five files by hand), after a pre-flight job checks the tag and
  runs the core tests.
- A single job creates the GitHub Release; wheels, sdist, C libraries and CLI
  binaries are attached to it. Nothing is published until every build succeeds.
- Added crates.io publishing of the `grav` crate.
- C library and CLI archives now also built for Linux aarch64 and macOS x86_64.
- Added `LICENSE-MIT` / `LICENSE-APACHE` and crate metadata.
