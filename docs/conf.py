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

"""
Sphinx documentation builder.
"""

import os
import sys
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
version = os.getenv("VERSION_STRING", "0.6")

# The full version, including alpha/beta/rc tags
release = os.getenv("RELEASE_STRING", "0.6.1")

project = "Qiskit Experiments"
copyright = f"2021-{datetime.date.today().year}, Qiskit Development Team"  # pylint: disable=redefined-builtin
author = "Qiskit Development Team"


# -- General configuration ---------------------------------------------------

extensions = [
    "qiskit_sphinx_theme",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
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
    "sphinx_remove_toctrees",
]

if os.getenv("PROD_BUILD", None):
    # Turn on view code source for production build
    extensions.append("sphinx.ext.viewcode")
else:
    # Remove stubs from the toctree for non-prod build because the full build is slow
    remove_from_toctrees = ["stubs/*"]


html_static_path = ["_static"]
templates_path = ["_templates"]
# Manually add the gallery CSS file for now
# TODO: Figure out why the styling is not working by default
html_css_files = ["nbsphinx-gallery.css", "customstyles.css"]

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
    "manuals/characterization/stark_experiment": "_images/stark_experiment_1_0.png",
    "**": "_static/images/logo.png",
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

# -- Options for HTML output -------------------------------------------------

html_theme = "qiskit-ecosystem"
html_title = f"{project} {release}"

docs_url_prefix = "qiskit-experiments"

html_last_updated_fmt = "%Y/%m/%d"

autoclass_content = "both"
intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "qiskit": ("https://docs.quantum.ibm.com/api/qiskit/", None),
    "uncertainties": ("https://pythonhosted.org/uncertainties", None),
    "pandas": ("http://pandas.pydata.org/docs/", None),
    "qiskit_aer": ("https://qiskit.github.io/qiskit-aer/", None),
    "qiskit_dynamics": ("https://qiskit-extensions.github.io/qiskit-dynamics/", None),
    "qiskit_ibm_runtime": ("https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/", None),
}


# Prepend warning for development docs:

if os.getenv("EXPERIMENTS_DEV_DOCS", None):
    rst_prolog = """
.. note::
    This is the documentation for the current state of the `development branch 
    <https://github.com/Qiskit-Extensions/qiskit-experiments/tree/main>`_
    of Qiskit Experiments. The documentation or APIs here can change prior to being
    released.
"""


def _get_versions(app, config):
    context = config.html_context
    start_version = (0, 5, 0)
    current_version = release
    current_version_info = current_version.split(".")
    if current_version_info[0] == "0":
        version_list = ["0.%s" % x for x in range(start_version[1], int(current_version_info[1]))]
    else:
        # TODO: When 1.0.0 add code to handle 0.x version list
        version_list = []
        pass
    context["version_list"] = version_list
    context["version_label"] = _get_version_label(current_version)


def _get_version_label(current_version):
    if not os.getenv("EXPERIMENTS_DEV_DOCS", None):
        return release
    else:
        return "Development"


def setup(app):
    app.connect("config-inited", _get_versions)
    app.connect("autodoc-skip-member", maybe_skip_member)


# Hardcoded list of class variables to skip in autodoc to avoid warnings
# Should come up with better way to address this

from qiskit_experiments.curve_analysis import ParameterRepr
from qiskit_experiments.curve_analysis import SeriesDef


def maybe_skip_member(app, what, name, obj, skip, options):
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
        "artifact_id",
        "artifact_data",
        "device_components",
        "created_time",
        "data",
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
