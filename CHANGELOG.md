# Changelog

## Unreleased

### CI
- R job installs testthat from the default (binary) repository and fails loudly if it is missing.

### Release pipeline
- Fixed the PyPI publish action (`pypa/gh-action-pypi-publish`).
- Release now verifies the tag against all package versions
  (`scripts/check_version.py`) and runs the core tests first.
- A single job creates the GitHub Release; wheels, sdist, C libraries and CLI
  binaries are attached to it. Nothing is published until every build succeeds.
- Added crates.io publishing of the `puddin` crate.
- C library and CLI archives now also built for Linux aarch64 and macOS x86_64.
- Added `LICENSE-MIT` / `LICENSE-APACHE` and crate metadata.
