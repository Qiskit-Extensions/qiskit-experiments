# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
A collection of utilities to generate documentation.
"""

import inspect
import re
import collections
from typing import List, Callable, Type, Iterator

from sphinx.config import Config as SphinxConfig
from sphinx.ext.napoleon.docstring import GoogleDocstring
from sphinx.util.docstrings import prepare_docstring

from qiskit_experiments.framework import BaseExperiment


_parameter_regex = re.compile(r"(.+?)\(\s*(.*[^\s]+)\s*\):(.*[^\s]+)")
_rest_role_regex = re.compile(r":(.+?) (.+?):\s*(.*[^\s]+)")


def _trim_empty_lines(docstring_lines: List[str]) -> List[str]:
    """A helper function to remove redundant line feeds."""
    i_start = 0
    lines_iter = iter(docstring_lines)
    while not next(lines_iter):
        i_start += 1

    i_end = len(docstring_lines)
    lines_iter = iter(docstring_lines[::-1])
    while not next(lines_iter):
        i_end -= 1

    return docstring_lines[i_start:i_end]


def _generate_analysis_ref(
    current_class: object,
    config: SphinxConfig = None,
    indent: str = "",
) -> List[str]:
    """Automatically generate analysis class reference with recursive ref to superclass."""

    if not issubclass(current_class, BaseExperiment):
        # check if no more base class
        raise TypeError("This is not valid experiment class.")

    experiment_option_parser = GoogleDocstring(
        docstring=prepare_docstring(current_class.__doc__, tabsize=len(indent)),
        config=config,
    )
    lines = list(map(lambda l: l.strip(), experiment_option_parser.lines()))

    analysis_ref_start = None
    try:
        analysis_ref_start = lines.index("# section: analysis_ref")
    except ValueError:
        super_classes = getattr(current_class, "__bases__")
        for super_cls in super_classes:
            try:
                return _generate_analysis_ref(
                    current_class=super_cls,
                    config=config,
                    indent=indent,
                )
            except Exception:
                pass

    if analysis_ref_start is None:
        raise Exception(f"Option docstring for analysis_ref is missing.")

    analysis_ref_lines = []
    for line in lines[analysis_ref_start + 1 :]:
        # add lines until hitting to next section
        if line.startswith("# section:"):
            break
        analysis_ref_lines.append(line)

    return analysis_ref_lines


def _check_no_indent(method: Callable) -> Callable:
    """Check indent of lines and return if this block is correctly indented."""

    def wraps(self, lines: List[str], *args, **kwargs):
        if all(l.startswith(" ") for l in lines):
            text_block = "\n".join(lines)
            raise ValueError(
                "Following documentation may have invalid indentation. "
                f"Please carefully check all indent levels are aligned. \n\n{text_block}"
            )
        return method(self, lines, *args, **kwargs)

    return wraps


def _get_superclass(current_class: Type, base_class: Type = None):
    """Get a list of restructured text of super classes of current class."""

    doc_classes = []
    mro_classes = inspect.getmro(current_class)[1:]
    if base_class:
        for mro_class in mro_classes:
            if issubclass(mro_class, base_class) and mro_class is not base_class:
                doc_classes.append(mro_class)
    else:
        doc_classes.extend(mro_classes)

    lines = []
    for doc_class in doc_classes:
        lines.append(f"* Superclass :class:`{doc_class.__module__}.{doc_class.__name__}`")

    return lines


def _write_options(lines, indent) -> Iterator:
    """A helper function to write options section.

    Consume restructured text of default options with role and create plain sphinx text.
    """

    prev_name = None
    params = collections.defaultdict(dict)
    tmp = {}
    for line in lines:
        if len(line) == 0 or line.isspace():
            continue
        matched = _rest_role_regex.match(line)
        if not matched:
            raise ValueError(
                f"{line} is not a valid directive. This must be parsed by docstring extension."
            )
        role = matched.group(1)
        name = matched.group(2)
        data = matched.group(3)
        if role == "mro_index":
            data = int(data)
        if prev_name and prev_name != name:
            params["Unknown class"][prev_name] = tmp
            tmp = {role: data}
            prev_name = name
        elif role == "source":
            params[data][name] = tmp
            tmp = {}
            prev_name = None
        else:
            tmp[role] = data
            prev_name = name

    if not params:
        yield "Option is not provided from this class."
    else:
        yield "Options"
        for source, data in params.items():
            yield indent + f"* Defined in the class {source}:"
            yield ""
            for name, info in data.items():
                _type = info.get("type", "n/a")
                _default = info.get("default_val", "n/a")
                _desc = info.get("param", "n/a")
                yield indent + f"  * **{name}** ({_type})"
                yield ""
                yield indent + f"    | Default value: {_default}"
                yield indent + f"    | {_desc}"
                yield ""
