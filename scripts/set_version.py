#!/usr/bin/env python3
"""Stamp one version into every package manifest.

The git tag (e.g. v0.2.0 or v0.2.0-rc1) is the source of truth for releases:
CI runs this at the start of every release job, so the numbers in the repo
only need to be touched by hand for local development.

Usage: set_version.py TAG_OR_VERSION
"""
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent

if len(sys.argv) != 2:
    sys.exit(__doc__)
version = sys.argv[1].removeprefix("v")
m = re.fullmatch(r"(\d+\.\d+\.\d+)(-[0-9A-Za-z.]+)?", version)
if not m:
    sys.exit(f"error: {version!r} is not a semantic version (X.Y.Z[-pre])")
numeric = m.group(1)  # R only allows numeric versions


def stamp(path, pattern, repl):
    p = root / path
    text = p.read_text()
    new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"error: no version found in {path}")
    p.write_text(new)
    print(f"{path:35} -> {numeric if path.endswith('DESCRIPTION') else version}")


# [workspace.package] / [package] / [project]: first `version = "..."` line.
for f in ("Cargo.toml", "bindings/python/Cargo.toml",
          "bindings/python/pyproject.toml", "bindings/julia/Project.toml"):
    stamp(f, r'^version = "[^"]*"', f'version = "{version}"')
stamp("bindings/r/DESCRIPTION", r"^Version:.*$", f"Version: {numeric}")
