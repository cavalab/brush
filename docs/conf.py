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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


import inspect
import importlib
from glob import glob
import subprocess
import os
import sys

sys.path.insert(0, os.path.abspath('../src'))
print("SPHINX SYSTEM PATH:")
print(sys.path)


def configureDoxyfile(input_dir, output_dir):
	with open('Doxyfile.in', 'r') as fp:
		filedata = fp.read()

	filedata = filedata.replace('@DOXYGEN_INPUT_DIR@', input_dir)
	filedata = filedata.replace('@DOXYGEN_OUTPUT_DIR@', output_dir)

	with open('Doxyfile', 'w') as fp2:
		fp2.write(filedata)


# Only trigger readthedocs build if running on readthedocs servers:
read_the_docs_build = os.environ.get('READTHEDOCS', None) == 'True'

breathe_projects = {}
# if read_the_docs_build:
if True:
	input_dir = '../src'
	output_dir = 'build'
	configureDoxyfile(input_dir, output_dir)
	subprocess.call('doxygen', shell=True)
	breathe_projects['brush'] = output_dir + '/xml'


# -- Project information -----------------------------------------------------

project = 'Brush'
copyright = '2021, William La Cava and Joseph D. Romano'
author = 'William La Cava and Joseph D. Romano'

# The full version, including alpha/beta/rc tags
release = '0.1a'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
	"sphinx.ext.autodoc",
	"sphinx.ext.autosummary",
	"breathe",  # Use Doxygen output as input for Sphinx
	"sphinx.ext.graphviz",
	"sphinx.ext.linkcode",
	'numpydoc',
	'sphinx_copybutton',
	'sphinx.ext.mathjax',
	'sphinx_math_dollar',
	'myst_nb',
	# 'nbsphinx',
	# "myst_parser",
]

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'thirdparty']

nb_output_stderr = "remove"
nb_execution_mode = "off"
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'
html_favicon = "_static/paint-brush-solid.svg"
# html_logo = "_static/paint-brush-solid.svg"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Breathe configuration
breathe_default_project = "brush"
breathe_default_members = ('members', 'undoc-members')
breathe_projects_source = {
	"brush": ("../src/", list(glob('../src/', recursive=True)))
}

html_theme_options = {
	"repository_url": "https://github.com/cavalab/brush",
	"use_repository_button": True,
	"path_to_docs": 'docs/',
	'home_page_in_toc': True,
	"show_navbar_depth": 1,
}

# linkcode resolution for sphinx.ext.linkcode
# adapted from https://github.com/readthedocs/sphinx-autoapi/issues/202
code_url = f"https://github.com/cavalab/brush/blob/master"

def linkcode_resolve(domain, info):
	if domain != 'py': 
		return None
	if not info['module']:
		return None

	mod = importlib.import_module(info["module"])
	if "." in info["fullname"]:
		objname, attrname = info["fullname"].split(".")
		obj = getattr(mod, objname)
		try:
			# object is a method of a class
			obj = getattr(obj, attrname)
		except AttributeError:
			# object is an attribute of a class
			return None
	else:
		obj = getattr(mod, info["fullname"])

	try:
		file = inspect.getsourcefile(obj)
		lines = inspect.getsourcelines(obj)
	except TypeError:
		# e.g. object is a typing.Union
		return None
	file = os.path.relpath(file, os.path.abspath(".."))
	start, end = lines[1], lines[1] + len(lines[0]) - 1

	return f"{code_url}/{file}#L{start}-L{end}"
