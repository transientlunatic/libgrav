#!/usr/bin/env python3
"""Finish the wasm-pack output in bindings/wasm/pkg so it can be published.

wasm-pack only knows about the wasm-bindgen output, so its generated
package.json lists just those files.  The TypeScript wrappers in
bindings/wasm/js (libgrav.ts, binary.ts) would otherwise be missing from the
npm tarball.  This compiles them next to the wasm output (JS + .d.ts) and adds
them, plus subpath exports, to package.json.

Run after `wasm-pack build --target bundler --out-dir pkg` in bindings/wasm:

    python3 scripts/finish_wasm_pkg.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

wasm = Path(__file__).resolve().parent.parent / "bindings/wasm"
pkg = wasm / "pkg"
modules = ["libgrav", "binary"]

if not (pkg / "package.json").exists():
    sys.exit(f"error: {pkg}/package.json not found; run wasm-pack build first")

tsc = shutil.which("tsc") or sys.exit("error: tsc not found (npm install -g typescript)")
# Compile from inside pkg/ so the wrappers can resolve ./grav_wasm.js
for m in modules:
    shutil.copy(wasm / "js" / f"{m}.ts", pkg / f"{m}.ts")
subprocess.run(
    [tsc, "--strict", "--declaration", "--target", "ES2022", "--module", "ESNext",
     "--moduleResolution", "bundler", *[f"{m}.ts" for m in modules]],
    check=True, cwd=pkg,
)

meta = json.loads((pkg / "package.json").read_text())
wasm_js = meta["main"]
meta["files"] = sorted(set(meta["files"]) | {f"{m}.{e}" for m in modules for e in ("js", "d.ts")})
meta["exports"] = {
    ".": {"types": f"./{meta['types']}", "import": f"./{wasm_js}"},
    **{f"./{m}": {"types": f"./{m}.d.ts", "import": f"./{m}.js"} for m in modules},
    "./package.json": "./package.json",
}
(pkg / "package.json").write_text(json.dumps(meta, indent=2) + "\n")
print("finished", pkg)
