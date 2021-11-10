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
# pylint: disable=method-hidden,too-many-return-statements,c-extension-no-member

"""Experiment serialization methods."""

import json
import math
import dataclasses
import importlib
import inspect
import warnings
import io
import base64
import zlib
import traceback
from types import FunctionType, MethodType
from typing import Any, Dict, Type, Optional, Union, Callable

import numpy as np
import scipy.sparse as sps

from qiskit.circuit import ParameterExpression, QuantumCircuit, qpy_serialization
from qiskit.circuit.library import BlueprintCircuit
from qiskit.result import Result
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit_experiments.version import __version__


def _show_warning(
    msg: Optional[str] = None,
    traceback_msg: Optional[str] = None,
    version: Optional[str] = None,
):
    """Show warning for partial deserialization"""
    warning_msg = "Returning partially deserialized value."
    if msg:
        warning_msg += f" {msg}"
    if version is not None and version != __version__:
        warning_msg += (
            f" NOTE: serialized object version ({version}) differs from the current "
            f" qiskit-experiments version ({__version__}."
        )
    if traceback_msg:
        warning_msg += f"The following exception was raised:\n{traceback_msg}"
    warnings.warn(warning_msg)


def _deprecation_warning(name: str, version: str):
    """Show warning for deprecated serialization"""
    warnings.warn(
        f"Derserializated data for <{name}> stored in a deprecated serialization format."
        " Re-serialize or re-save the data to update the serialization format otherwise"
        f" loading this data may fail in a qiskit-experiments version {version}",
        DeprecationWarning,
    )


def _serialize_bytes(data: bytes, compress: bool = True) -> str:
    """Serialize binary data.

    Args:
        data: Data to be serialized.
        compress: Whether to compress the serialized data.

    Returns:
        String representation.
    """
    if compress:
        data = zlib.compress(data)
    value = {
        "encoded": base64.standard_b64encode(data).decode("utf-8"),
        "compressed": compress,
    }
    return {"__type__": "b64encoded", "__value__": value}


def _deserialize_bytes(value: Dict) -> str:
    """Deserialize binary encoded data.

    Args:
        data: Data to be serialized.

    Returns:
        String representation.
    """
    try:
        encoded = value["encoded"]
        compressed = value["compressed"]
        decoded = base64.standard_b64decode(encoded)
        if compressed:
            decoded = zlib.decompress(decoded)
        return decoded
    except Exception as ex:  # pylint: disable=broad-except
        warning_msg = "Could not deserialize binary encoded data."
        traceback_msg = traceback.format_exception(type(ex), ex, ex.__traceback__)
        _show_warning(warning_msg, traceback_msg=traceback_msg)
        return value


def _serialize_and_encode(
    data: Any, serializer: Callable, compress: bool = True, **kwargs: Any
) -> str:
    """Serialize the input data and return the encoded string.

    Args:
        data: Data to be serialized.
        serializer: Function used to serialize data.
        compress: Whether to compress the serialized data.
        kwargs: Keyword arguments to pass to the serializer.

    Returns:
        String representation.
    """
    buff = io.BytesIO()
    serializer(buff, data, **kwargs)
    buff.seek(0)
    serialized_data = buff.read()
    buff.close()
    value = _serialize_bytes(serialized_data, compress=compress)
    return value


def _decode_and_deserialize(value: Dict, deserializer: Callable, name: Optional[str] = None) -> Any:
    """Decode and deserialize input data.

    Args:
        value: The binary encoded serialized data value.
        deserializer: Function used to deserialize data.
        name: Object type name for warning message if deserialization fails.

    Returns:
        Deserialized data.
    """
    try:
        buff = io.BytesIO()
        buff.write(value)
        buff.seek(0)
        orig = deserializer(buff)
        buff.close()
        return orig
    except Exception as ex:  # pylint: disable=broad-except
        warning_msg = f"Could not deserialize <{name}> data."
        traceback_msg = traceback.format_exception(type(ex), ex, ex.__traceback__)
        _show_warning(warning_msg, traceback_msg=traceback_msg)
        return value


def _serialize_safe_float(obj: any):
    """Recursively serialize basic types safely handing inf and NaN"""
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        else:
            value = obj
            if math.isnan(obj):
                value = "NaN"
            elif obj == math.inf:
                value = "Infinity"
            elif obj == -math.inf:
                value = "-Infinity"
            return {"__type__": "safe_float", "__value__": value}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_safe_float(i) for i in obj]
    elif isinstance(obj, dict):
        return {key: _serialize_safe_float(val) for key, val in obj.items()}
    elif isinstance(obj, complex):
        return {"__type__": "complex", "__value__": _serialize_safe_float([obj.real, obj.imag])}
    return obj


def _serialize_object(obj: Any, settings: Optional[Dict] = None, safe_float: bool = True) -> Dict:
    """Serialize a class instance from its init args and kwargs.

    Args:
        obj: The object to be serialized.
        settings: Optional, settings for reconstructing the object from kwargs.
        safe_float: if True check float values for NaN, inf and -inf
                    and cast to strings during serialization.

    Returns:
        Dict serialized class instance.
    """
    value = {
        "name": type(obj).__name__,
        "module": type(obj).__module__,
        "version": __version__,
    }
    if settings is None:
        if hasattr(obj, "__json_encode__"):
            settings = obj.__json_encode__()
        elif hasattr(obj, "settings"):
            settings = obj.settings
        else:
            settings = {}
    if safe_float:
        settings = _serialize_safe_float(settings)
    value["settings"] = settings
    return {"__type__": "object", "__value__": value}


def _deserialize_object(value: Dict) -> Any:
    """Deserialize class instance saved as settings"""
    name = value["name"]
    mod = value["module"]
    version = value.get("version", None)
    settings = value.get("settings", {})

    cls = None
    if mod == "__main__":
        cls = globals().get(name, None)
    else:
        scope = importlib.import_module(mod)
        for name_, obj in inspect.getmembers(scope, inspect.isclass):
            if name_ == name:
                cls = obj
                break

    # Warning msg if deserialization fails
    traceback_msg = None
    warning_msg = None
    if cls is None:
        warning_msg = f"Cannot deserialize {name}. The type could not be found in module {mod}"
    elif hasattr(cls, "__json_decode__"):
        try:
            return cls.__json_decode__(settings)
        except Exception as ex:  # pylint: disable=broad-except
            traceback_msg = traceback.format_exception(type(ex), ex, ex.__traceback__)
            warning_msg = (
                f"Could not deserialize instance of class {name} from value {settings} "
                "using __json_decode__ method."
            )
    else:
        try:
            return cls(**settings)
        except Exception as ex:  # pylint: disable=broad-except
            traceback_msg = traceback.format_exception(type(ex), ex, ex.__traceback__)
            warning_msg = (
                f"Could not deserialize instance of class {name} from settings {settings}."
            )

    # Display warning msg if deserialization failed
    _show_warning(warning_msg, traceback_msg=traceback_msg, version=version)

    # Return partially deserialized value
    return value


def is_type(obj: Any) -> bool:
    """Return True if object is a class, function, or method type"""
    return inspect.isclass(obj) or inspect.isfunction(obj) or inspect.ismethod(obj)


def _serialize_type(type_name: Union[Type, FunctionType, MethodType]):
    """Serialize a type, function, or class method"""
    value = {
        "name": type_name.__qualname__,
        "module": type_name.__module__,
        "version": __version__,
    }
    return {"__type__": "type", "__value__": value}


def _deserialize_type(value: Dict):
    """Deserialize a type, function, or class method"""
    traceback_msg = None
    try:
        version = value.get("version", None)
        qualname = value["name"].split(".", maxsplit=1)
        if len(qualname) == 2:
            method_cls, name = qualname
        else:
            method_cls = None
            name = qualname[0]
        mod = value["module"]

        scope = None
        if mod == "__main__":
            if method_cls is None:
                if name in globals():
                    return globals()[name]
            else:
                scope = globals().get(method_cls, None)
        else:
            mod_scope = importlib.import_module(mod)
            if method_cls is None:
                scope = mod_scope
            else:
                for name_, obj in inspect.getmembers(mod_scope, inspect.isclass):
                    if name_ == method_cls:
                        scope = obj

        if scope is not None:
            for name_, obj in inspect.getmembers(scope, is_type):
                if name_ == name:
                    return obj
    except Exception as ex:  # pylint: disable=broad-except
        traceback_msg = traceback.format_exception(type(ex), ex, ex.__traceback__)

    # Show warning
    warning_msg = f"Cannot deserialize {name}. The type could not be found in module {mod}"
    _show_warning(warning_msg, traceback_msg=traceback_msg, version=version)

    # Return partially deserialized value
    return value


def _deserialize_object_legacy(value: Dict) -> Any:
    """Deserialize a class object from its init args and kwargs."""
    try:
        class_name = value["__name__"]
        mod_name = value["__module__"]
        args = value.get("__args__", tuple())
        kwargs = value.get("__kwargs__", dict())
        mod = importlib.import_module(mod_name)
        for name, cls in inspect.getmembers(mod, inspect.isclass):
            if name == class_name:
                return cls(*args, **kwargs)

        raise Exception(f"Unable to find class {class_name} in module {mod_name}")

    except Exception as ex:  # pylint: disable=broad-except
        traceback_msg = traceback.format_exception(type(ex), ex, ex.__traceback__)
        warning_msg = f"Unable to initialize {class_name}."
        _show_warning(warning_msg, traceback_msg=traceback_msg)
        return value


class ExperimentEncoder(json.JSONEncoder):
    """JSON Encoder for Numpy arrays and complex numbers."""

    def default(self, obj: Any) -> Any:  # pylint: disable=arguments-differ
        if isinstance(obj, complex):
            return _serialize_safe_float(obj)
        if isinstance(obj, set):
            return {"__type__": "set", "__value__": list(obj)}
        if isinstance(obj, np.ndarray):
            value = _serialize_and_encode(obj, np.save, allow_pickle=False)
            return {"__type__": "ndarray", "__value__": value}
        if isinstance(obj, sps.spmatrix):
            value = _serialize_and_encode(obj, sps.save_npz, compress=False)
            return {"__type__": "spmatrix", "__value__": value}
        if isinstance(obj, bytes):
            return _serialize_bytes(obj)
        if dataclasses.is_dataclass(obj):
            return _serialize_object(obj, settings=dataclasses.asdict(obj))
        if isinstance(obj, QuantumCircuit):
            # TODO Remove the decompose when terra 6713 is released.
            if isinstance(obj, BlueprintCircuit):
                obj = obj.decompose()
            value = _serialize_and_encode(
                data=obj, serializer=lambda buff, data: qpy_serialization.dump(data, buff)
            )
            return {"__type__": "QuantumCircuit", "__value__": value}
        if isinstance(obj, ParameterExpression):
            value = _serialize_and_encode(
                data=obj,
                serializer=qpy_serialization._write_parameter_expression,
                compress=False,
            )
            return {"__type__": "ParameterExpression", "__value__": value}
        if isinstance(obj, Result):
            return {"__type__": "Result", "__value__": obj.to_dict()}
        if isinstance(obj, QuantumChannel):
            # Temporary fix for incorrect settings in qiskit-terra
            # See https://github.com/Qiskit/qiskit-terra/pull/7194
            settings = {
                "data": obj.data,
                "input_dims": obj.input_dims(),
                "output_dims": obj.output_dims(),
            }
            return _serialize_object(obj, settings=settings)
        if isinstance(obj, DensityMatrix):
            # Temporary fix for incorrect settings in qiskit-terra
            # See https://github.com/Qiskit/qiskit-terra/pull/7194
            settings = {
                "data": obj.data,
                "dims": obj.dims(),
            }
            return _serialize_object(obj, settings=settings)
        if is_type(obj):
            return _serialize_type(obj)
        try:
            return super().default(obj)
        except TypeError:
            return _serialize_object(obj)


class ExperimentDecoder(json.JSONDecoder):
    """JSON Decoder for Numpy arrays and complex numbers."""

    _NaNs = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        """Object hook."""
        if "__type__" in obj:
            obj_type = obj["__type__"]
            obj_val = obj["__value__"]
            if obj_type == "complex":
                return obj_val[0] + 1j * obj_val[1]
            if obj_type == "ndarray":
                return _decode_and_deserialize(obj_val, np.load, name=obj_type)
            if obj_type == "spmatrix":
                return _decode_and_deserialize(obj_val, sps.load_npz, name=obj_type)
            if obj_type == "b64encoded":
                return _deserialize_bytes(obj_val)
            if obj_type == "set":
                return set(obj_val)
            if obj_type == "QuantumCircuit":
                return _decode_and_deserialize(obj_val, qpy_serialization.load, name=obj_type)[0]
            if obj_type == "ParameterExpression":
                return _decode_and_deserialize(
                    obj_val, qpy_serialization._read_parameter_expression, name=obj_type
                )
            if obj_type == "Result":
                return Result.from_dict(obj_val)
            if obj_type == "safe_float":
                return self._NaNs.get(obj_val, obj_val)
            if obj_type == "object":
                return _deserialize_object(obj_val)
            if obj_type == "type":
                return _deserialize_type(obj_val)

            # Deprecated formats
            if obj_type == "array":
                _deprecation_warning(obj_type, "0.3.0")
                return np.array(obj_val)
            if obj_type == "function":
                _deprecation_warning(obj_type, "0.3.0")
                return obj_val
            if obj_type == "__object__":
                _deprecation_warning(obj_type, "0.3.0")
                return _deserialize_object_legacy(obj_val)
            if obj_type == "__class_name__":
                _deprecation_warning(obj_type, "0.3.0")
                return obj_val
        return obj
