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


# -- Project information -----------------------------------------------------

project = "Bagua API Documentation"
copyright = "2021, Kuaishou AI Platform and DS3 Lab"
author = "Kuaishou AI Platform and DS3 Lab"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]


napoleon_numpy_docstring = True
autoapi_python_class_content = "both"
autodoc_typehints = "description"
autoapi_type = "python"
autoapi_dirs = ["../bagua"]
autoapi_root = "autoapi"
autoapi_template_dir = "_autoapi_templates"
autoapi_ignore = [
    "*/bagua/autotune/*",
    "*/bagua/distributed/*",
    "*/bagua/script/*",
    "*/bagua/service/*",
    "*/bagua/torch_api/tools/*.py",
    "*/bagua/torch_api/exceptions.py",
    "*/bagua/torch_api/utils.py",
    "*/bagua/torch_api/globals.py",
    "*/bagua/version.py",
    "*/bagua/bagua_define.py",
]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "imported-members",
]
autoapi_member_order = "groupwise"

master_doc = "autoapi/index"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = True

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "bagua-sphinx-theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "show_powered_by": False,
    "github_user": "BaguaSys",
    "github_repo": "bagua",
    "github_banner": True,
    "show_related": False,
    "note_bg": "#FFF59C",
}

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True


_ignore_methods = [
    "bagua.torch_api.contrib.LoadBalancingDistributedSampler.shuffle_chunks",
    "bagua.torch_api.contrib.LoadBalancingDistributedBatchSampler.generate_batches",
    "bagua.torch_api.contrib.load_balancing_data_loader.LoadBalancingDistributedSampler.shuffle_chunks",
    "bagua.torch_api.contrib.load_balancing_data_loader.LoadBalancingDistributedBatchSampler.generate_batches",
]
_ignore_functions = [
    "bagua.torch_api.env.get_autotune_server_addr",
    "bagua.torch_api.env.get_master_addr",
    "bagua.torch_api.env.get_bagua_service_port",
    "bagua.torch_api.env.is_report_metrics_switch_on",
    "bagua.torch_api.env.get_autotune_level",
    "bagua.torch_api.env.get_autotune_max_samples",
    "bagua.torch_api.env.get_autotune_sampling_confidence_time_s",
    "bagua.torch_api.env.get_autotune_warmup_time_s",
    "bagua.torch_api.env.get_is_output_autotune_log",
    "bagua.torch_api.globals.is_initialized",
    "bagua.torch_api.communication.get_bagua_hyperparameters",
    "bagua.torch_api.communication.get_hyperparameters_service_client",
    "bagua.torch_api.communication.gen_nccl_unique_id",
    "bagua.torch_api.communication.init_bagua_inter_communicator",
    "bagua.torch_api.communication.init_bagua_intra_communicator",
    "bagua.torch_api.communication.init_bagua_communicator",
    "bagua.torch_api.communication.broadcast_coalesced",
    "bagua.torch_api.communication.allreduce_coalesced",
    "bagua.torch_api.communication.start_autotune_server",
    "bagua.torch_api.communication.run_flask_app",
]
_ignore_classes = [
    "bagua.torch_api.communication.BaguaGlobalState",
    "bagua.torch_api.algorithms.BaguaModule",
    "bagua.torch_api.algorithms.BaguaBucket",
    "bagua.torch_api.algorithms.BaguaTensor",
]


def skip_methods(app, what, name, obj, skip, options):
    if what == "method" and name in _ignore_methods:
        skip = True
        return skip

    if what == "function" and name in _ignore_functions:
        skip = True
        return skip

    if what == "class" and name in _ignore_classes:
        skip = True
        return skip

    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_methods)
