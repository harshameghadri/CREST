"""Sphinx configuration for the CREST documentation (Read the Docs and local builds).

Build locally:  pip install -r docs/requirements.txt && sphinx-build -b html docs docs/_build/html
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # import crest from the source tree; no Rust build needed

# -- project -------------------------------------------------------------------------------
project = "CREST"
author = "harshameghadri"
copyright = "2026, harshameghadri"
_cargo = (ROOT / "Cargo.toml").read_text()
release = re.search(r'^version\s*=\s*"([^"]+)"', _cargo, re.M).group(1)  # single source of truth
version = release

# -- extensions ----------------------------------------------------------------------------
extensions = [
    "myst_parser",            # Markdown pages
    "sphinx.ext.autodoc",     # API reference from docstrings
    "sphinx.ext.napoleon",    # NumPy-style "Parameters" sections
    "sphinx.ext.viewcode",    # [source] links
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "requirements.txt"]

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "fieldlist"]
myst_heading_anchors = 3

# The compiled extension (crest/crest.*.so) is not built on Read the Docs. Mock it, and the
# optional heavy dependencies that are imported lazily, so autodoc can import every module.
autodoc_mock_imports = ["crest.crest", "h5py", "scipy", "anndata", "pandas", "sklearn"]
autodoc_default_options = {"members": True, "member-order": "bysource", "show-inheritance": False}
autodoc_typehints = "description"
autodoc_preserve_defaults = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# links to GitHub files mentioned in the Markdown pages
myst_url_schemes = {
    "http": None, "https": None, "mailto": None,
    "gh": "https://github.com/harshameghadri/CREST/blob/dev/{{path}}",
}

# -- HTML ----------------------------------------------------------------------------------
html_theme = "furo"
html_title = f"CREST {release}"
html_theme_options = {
    "source_repository": "https://github.com/harshameghadri/CREST/",
    "source_branch": "dev",
    "source_directory": "docs/",
    "light_css_variables": {"color-brand-primary": "#2a78d6", "color-brand-content": "#2a78d6"},
    "dark_css_variables": {"color-brand-primary": "#6ea8ef", "color-brand-content": "#6ea8ef"},
}
copybutton_prompt_text = r"^\$ |^>>> "
copybutton_prompt_is_regexp = True

# existing design notes cross-reference files outside docs/ (bench/, tests/); those are shown
# as code, not links, so these warnings are expected and harmless
suppress_warnings = ["myst.xref_missing", "myst.header"]
