# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Functions for checking dependency versions.
"""

from importlib.metadata import version
import sys


def numpy_version():
    """Returns the current numpy version in (major, minor) form."""
    return tuple(map(int, version("numpy").split(".")[:2]))


def python_version():
    """Returns the current python version in (major, minor) form."""
    return tuple(map(int, sys.version.split(".")[:2]))
