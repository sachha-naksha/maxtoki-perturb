"""Sphinx config for maxtoki-mlx torch_pipeline docs."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the source tree importable for autodoc.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

project = "maxtoki-mlx torch_pipeline"
author = "Srijit Iyer"
copyright = "2026, Srijit Iyer"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "myst_parser",
]

# Build cleanly even when heavyweight scientific deps are missing on the
# docs host (numpy / torch / anndata / datasets / bionemo). Autodoc skips
# imports that fail and falls back to module-level introspection.
autodoc_mock_imports = [
    "numpy",
    "scipy",
    "torch",
    "anndata",
    "scanpy",
    "datasets",
    "bionemo",
    "nemo",
    "megatron",
    "pytorch_lightning",
    "pandas",
    "yaml",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_attr_annotations = True

autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns: list[str] = []

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Read the Docs theme. ``furo`` is also fine if you prefer; just swap below.
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Don't fail on missing references inside docstrings
nitpicky = False
