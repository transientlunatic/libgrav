# Configuration file for the Sphinx documentation builder.

project = "libgrav"
copyright = "2026, The libgrav Authors"
author = "The libgrav Authors"
release = "0.1.0"

extensions = [
    "sphinx_rust",
    "myst_parser",
    "sphinxcontrib.katex",
]

# sphinx-rust: paths to Cargo.toml of each crate to document
rust_crates = [
    "../crates/grav",
]

# MyST
myst_enable_extensions = ["dollarmath", "amsmath"]

# HTML theme
html_theme = "furo"
html_title = "libgrav"

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst",
}
