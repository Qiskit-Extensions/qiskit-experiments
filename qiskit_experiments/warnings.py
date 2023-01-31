# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A collection of typical warnings."""

import functools
import warnings
from typing import Callable, Optional, Type

from qiskit import QiskitError


def deprecated_function(
    last_version: Optional[str] = None,
    msg: Optional[str] = None,
    stacklevel: int = 2,
) -> Callable:
    """A function or method decorator to show deprecation warning.

    Args:
        last_version: The Qiskit Experiment version that this function is removed.
        msg: Extra message, for example, to indicate alternative approach.
        stacklevel: Stacklevel of this warning. See Python Warnings documentation for details.

    Examples:

        .. code-block::

            @deprecated_function(last_version="0.3", msg="Use new_function instead.")
            def old_function(*args, **kwargs):
                pass

            def new_function(*args, **kwargs):
                pass

    Returns:
        Deprecated function or method.
    """

    def deprecated_wrapper(func: Callable):
        @functools.wraps(func)
        def _wrap(*args, **kwargs):
            namespace = func.__qualname__.split(".")
            if len(namespace) == 1:
                message = f"The function '{func.__name__}' has been deprecated and "
            else:
                cls_name, meth_name = namespace
                message = f"The method '{meth_name}' of '{cls_name}' class has been deprecated and "
            if last_version:
                message += f"will be removed after Qiskit Experiments {last_version}. "
            else:
                message += "will be removed in future release. "
            if msg:
                message += msg
            warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)
            return func(*args, **kwargs)

        return _wrap

    return deprecated_wrapper


def deprecated_class(
    last_version: Optional[str] = None,
    new_cls: Optional[Type] = None,
    msg: Optional[str] = None,
    stacklevel: int = 2,
) -> Callable:
    """A class decorator to show deprecation warning and
    patch __new__ method of the class to instantiate the new class.

    Args:
        last_version: The last Qiskit Experiments version that will have this class.
        new_cls: Alternative class type.
        msg: Extra message, for example, to indicate alternative approach.
        stacklevel: Stacklevel of this warning. See Python Warnings documentation for details.

    Examples:

        .. code-block::

            @deprecated_class(last_version="0.3", new_cls=NewCls)
            class OldClass:
                pass

            class NewClass:
                pass

    Returns:
        Deprecated class.
    """

    def patch_new(cls) -> Type:
        @functools.wraps(cls.__init__, assigned=("__annotations__",))
        def new(deprecated_cls, *args, **kwargs):
            message = f"Class '{deprecated_cls.__name__}' has been deprecated"
            if new_cls:
                message += f" and replaced with '{new_cls.__name__}'."
            else:
                message += ". "
            if last_version:
                message += f"This class will be removed after Qiskit Experiments {last_version}."
            else:
                message += "This class will be removed in a future release."
            message += f"The '{deprecated_cls.__name__}' instance cannot be loaded after removal."
            if msg:
                message += msg
            warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)
            instance = object.__new__(new_cls or deprecated_cls)
            instance.__init__(*args, **kwargs)
            return instance

        cls.__new__ = new
        return cls

    return patch_new


def deprecate_arguments(kwarg_map, category: Type[Warning] = DeprecationWarning):
    """Decorator to automatically alias deprecated argument names and warn upon use."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if kwargs:
                _rename_kwargs(func.__name__, kwargs, kwarg_map, category)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _rename_kwargs(func_name, kwargs, kwarg_map, category: Type[Warning] = DeprecationWarning):
    for old_arg, new_arg in kwarg_map.items():
        if old_arg in kwargs:
            if new_arg in kwargs:
                raise TypeError(f"{func_name} received both {new_arg} and {old_arg} (deprecated).")

            if new_arg is None:
                warnings.warn(
                    f"{func_name} keyword argument {old_arg} is deprecated and "
                    "will in future be removed.",
                    category=category,
                    stacklevel=3,
                )
            else:
                if new_arg in kwargs:
                    raise QiskitError(
                        f"{func_name} received both deprecated key argument {old_arg}"
                        " and new key argument {new_arg}"
                    )

                warnings.warn(
                    f"{func_name} keyword argument {old_arg} is deprecated and "
                    f"replaced with {new_arg}.",
                    category=category,
                    stacklevel=3,
                )

                kwargs[new_arg] = kwargs.pop(old_arg)

                
def specific_deprecate():
    """Decorator to deprecate from qubit to physcial_qubits"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            category = DeprecationWarning
            func_name = args[0].__class__.__name__ + ".__init__"
            
            if len(args) > 1 and isinstance(args[1], int):
                args = list(args)
                args[1] = [args[1]]
                args = tuple(args)
                warnings.warn(f"The first argument of {func_name} has been renamed from qubits to "
                              "physical_qubits, and is expecting a sequence instead of an integer. "
                              "Support of integer values is deprecated and will be removed.",
                              category=category,
                              stacklevel=3
                              )
                
            if kwargs and "qubit" in kwargs:
                if "physical_qubits" in kwargs:
                    raise TypeError(f"{func_name} received both physical_qubits and qubits "
                                    "(deprecated).")

                warnings.warn(
                    f"{func_name} keyword argument qubit is deprecated and "
                    f"replaced with physical_qubits.",
                    category=category,
                    stacklevel=3,
                )

                kwargs["physical_qubits"] = [kwargs.pop("qubit")]

            return func(*args, **kwargs)

        return wrapper

    return decorator
