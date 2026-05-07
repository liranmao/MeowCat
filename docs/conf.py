import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath('..'))

project = "MeowCat"
author = "Liran Mao, Mingyao Li"
copyright = f"Mingyao Li Lab, {datetime.now().year}"
release = "0.1.0, 2025"

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinxarg.ext',
]

autosummary_generate = False
add_module_names = False
html_show_sourcelink = False

templates_path = ['_templates']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = "_static/logo.png"
html_css_files = ["readthedocs-custom.css"]
html_theme_options = {
    'logo_only': True,
}
