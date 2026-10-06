# Changelog

## Unreleased

### CI
- R job installs testthat from the default (binary) repository and fails loudly if it is missing.

### Release pipeline
- Fixed the PyPI publish action (`pypa/gh-action-pypi-publish`).
- The git tag is the single source of truth for the version: every release job
  stamps it into all manifests with `scripts/set_version.py` (so there is no
  need to bump five files by hand), after a pre-flight job checks the tag and
  runs the core tests.
- A single job creates the GitHub Release; wheels, sdist, C libraries and CLI
  binaries are attached to it. Nothing is published until every build succeeds.
- Added crates.io publishing of the `puddin` crate.
- C library and CLI archives now also built for Linux aarch64 and macOS x86_64.
- Added `LICENSE-MIT` / `LICENSE-APACHE` and crate metadata.
