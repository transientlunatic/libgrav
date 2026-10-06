#!/usr/bin/env python3
"""Check that a release tag (e.g. v0.2.0) matches every package version.

Usage: check_version.py [TAG]   (TAG defaults to no tag check)
"""
import re
import sys
import tomllib
from pathlib import Path

root = Path(__file__).resolve().parent.parent


def toml(path):
    with open(root / path, "rb") as f:
        return tomllib.load(f)


versions = {
    "Cargo.toml (workspace)": toml("Cargo.toml")["workspace"]["package"]["version"],
    "bindings/python/Cargo.toml": toml("bindings/python/Cargo.toml")["package"]["version"],
    "bindings/python/pyproject.toml": toml("bindings/python/pyproject.toml")["project"]["version"],
    "bindings/julia/Project.toml": toml("bindings/julia/Project.toml")["version"],
}
m = re.search(r"^Version:\s*(\S+)", (root / "bindings/r/DESCRIPTION").read_text(), re.M)
versions["bindings/r/DESCRIPTION"] = m.group(1)

if len(sys.argv) > 1:
    versions["git tag"] = sys.argv[1].removeprefix("v")

for k, v in versions.items():
    print(f"{k:35} {v}")
if len(set(versions.values())) != 1:
    sys.exit("error: versions disagree")
print("ok")
