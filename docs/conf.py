# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = 'BFit'
copyright = '2021, The QC-Devs Community'
author = 'The QC-Devs Community'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'numpydoc',
    # 'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    # for adding “copy to clipboard” buttons to all text/code boxes
    'sphinx_copybutton',
    'autoapi.extension',
    'nbsphinx',
    #'sphinxcontrib.bibtex'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# Autoapi options
# can be found in https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html
autoapi_generate_api_docs = True
autoapi_type = 'python'
autoapi_dirs = ['../bfit/']
autoapi_ignore = ["*/test_*.py"]
autoapi_python_class_content = "both"
autodoc_member_order = 'bysource'
autoapi_options = [
    "members", "inherited-members", "undoc-members", "show-inheritance",
    "show-module-summary",
]


# Napolean settings to allow documentation of __init__
napoleon_include_special_with_doc = True
napoleon_include_init_with_doc = True
