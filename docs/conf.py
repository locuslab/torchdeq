# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TorchDEQ'
copyright = '2023, TorchDEQ'
author = 'Zhengyang Geng'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    # 'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    # 'sphinx.ext.viewcode',
    'sphinxcontrib.katex',
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton',
    # 'sphinx_panels',
    'myst_parser',
]


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]

copybutton_prompt_text = r'>>> '
copybutton_prompt_is_regexp = True

