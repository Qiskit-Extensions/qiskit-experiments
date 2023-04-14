# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name, wrong-import-position, unused-argument

"""
Sphinx documentation builder.
"""

import os
import sys
import subprocess
import datetime

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("./_ext"))
sys.path.append(os.path.abspath(".."))

# Set env flag so that we can doc functions that may otherwise not be loaded
# see for example interactive visualizations in qiskit.visualization.
os.environ["QISKIT_DOCS"] = "TRUE"

# -- Project information -----------------------------------------------------
# The short X.Y version
version = "0.6"
# The full version, including alpha/beta/rc tags
release = "0.6.0"
project = f"Qiskit Experiments {version}"
copyright = f"2021-{datetime.date.today().year}, Qiskit Development Team"  # pylint: disable=redefined-builtin
author = "Qiskit Development Team"


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx_copybutton",
    "jupyter_sphinx",
    "reno.sphinxext",
    "sphinx_design",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "autoref",
    "autodoc_experiment",
    "autodoc_analysis",
    "autodoc_visualization",
    "jupyter_execute_custom",
]

html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["gallery.css"]

nbsphinx_timeout = 360
nbsphinx_execute = os.getenv("QISKIT_DOCS_BUILD_TUTORIALS", "never")
nbsphinx_widgets_path = ""
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# Thumbnails for experiment manuals from output images
# These should ideally be automatically generated using a custom macro to specify
# chosen cells for thumbnails, like the nbsphinx-gallery tag
nbsphinx_thumbnails = {
    "manuals/verification/quantum_volume": "_images/quantum_volume_2_0.png",
    "manuals/measurement/readout_mitigation": "_images/readout_mitigation_4_0.png",
    "manuals/verification/randomized_benchmarking": "_images/randomized_benchmarking_3_1.png",
    "manuals/measurement/restless_measurements": "_images/restless_shots.png",
    "manuals/verification/state_tomography": "_images/state_tomography_3_0.png",
    "manuals/characterization/t1": "_images/t1_0_0.png",
    "manuals/characterization/t2ramsey": "_images/t2ramsey_4_0.png",
    "manuals/characterization/tphi": "_images/tphi_5_1.png",
    "manuals/characterization/t2hahn": "_images/t2hahn_5_0.png",
    "**": "_static/no_image.png",
}

# Add `data keys` and `style parameters` alias. Needed for `expected_*_data_keys` methods in
# visualization module and `default_style` method in `PlotStyle` respectively.
napoleon_custom_sections = [("data keys", "params_style"), ("style parameters", "params_style")]

# Move type hints from signatures to the parameter descriptions (except in overload cases, where
# that's not possible).
autodoc_typehints = "description"
# Only add type hints from signature to description body if the parameter has documentation.  The
# return type is always added to the description (if in the signature).
autodoc_typehints_description_target = "documented_params"

autosummary_generate = True

autodoc_default_options = {"inherited-members": None}

# If true, figures, tables and code-blocks are automatically numbered if they
# have a caption.
numfig = True

# A dictionary mapping 'figure', 'table', 'code-block' and 'section' to
# strings that are used for format of figure numbers. As a special character,
# %s will be replaced to figure number.
numfig_format = {"table": "Table %s"}

language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "colorful"

# A boolean that decides whether module names are prepended to all object names
# (for object types where a “module” of some kind is defined), e.g. for
# py:function directives.
add_module_names = False

# A list of prefixes that are ignored for sorting the Python module index
# (e.g., if this is set to ['foo.'], then foo.bar is shown under B, not F).
# This can be handy if you document a project that consists of a single
# package. Works only for the HTML builder currently.
modindex_common_prefix = ["qiskit_experiments."]

# -- Configuration for extlinks extension ------------------------------------
# Refer to https://www.sphinx-doc.org/en/master/usage/extensions/extlinks.html


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "qiskit_sphinx_theme"  # use the theme in subdir 'theme'

html_context = {
    "analytics_enabled": True,
    "expandable_sidebar": True,
}

html_last_updated_fmt = "%Y/%m/%d"

html_theme_options = {
    "logo_only": True,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
}


autoclass_content = "both"
intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "qiskit": ("https://qiskit.org/documentation/", None),
    "uncertainties": ("https://pythonhosted.org/uncertainties", None),
    "qiskit_ibm_provider": ("https://qiskit.org/documentation/partners/qiskit_ibm_provider", None),
}


# Prepend warning for development docs:

if os.getenv("EXPERIMENTS_DEV_DOCS", None):
    rst_prolog = """
.. note::
    This is the documentation for the current state of the development branch
    of Qiskit Experiments. The documentation or APIs here can change prior to being
    released.
"""


def _get_versions(app, config):
    context = config.html_context
    start_version = (0, 5, 0)
    proc = subprocess.run(["git", "describe", "--abbrev=0"], capture_output=True, check=False)
    proc.check_returncode()
    current_version = proc.stdout.decode("utf8")
    current_version_info = current_version.split(".")
    if current_version_info[0] == "0":
        version_list = list(
            f"0.{x}" for x in range(start_version[1], int(current_version_info[1]) + 1)
        )
    else:
        # TODO: When 1.0.0 add code to handle 0.x version list
        version_list = []
        pass
    context["version_list"] = version_list
    context["version_label"] = _get_version_label()


def _get_version_label():
    if not os.getenv("EXPERIMENTS_DEV_DOCS", None):
        return release
    else:
        return "Development"


def setup(app):
    """Set up plugins."""
    app.connect("config-inited", _get_versions)
    app.connect("autodoc-skip-member", maybe_skip_member)


from qiskit_experiments.curve_analysis import ParameterRepr
from qiskit_experiments.curve_analysis import SeriesDef


def maybe_skip_member(app, what, name, obj, skip, options):
    """Hardcoded list of class variables to skip in autodoc to avoid warnings.
    TODO: Fix class attribute generation to avoid hardcoding.
    """
    skip_names = [
        "analysis",
        "set_run_options",
        "data_allocation",
        "labels",
        "shots",
        "x",
        "y",
        "y_err",
        "name",
        "filter_kwargs",
        "fit_func",
        "signature",
    ]
    skip_members = [
        ParameterRepr.repr,
        ParameterRepr.unit,
        SeriesDef.plot_color,
        SeriesDef.plot_symbol,
        SeriesDef.model_description,
        SeriesDef.canvas,
    ]
    if not skip:
        return (name in skip_names or obj in skip_members) and what == "attribute"
    return skip
