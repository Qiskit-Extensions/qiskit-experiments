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
# pylint: disable=method-hidden,too-many-return-statements

"""Experiment serialization methods."""

import json
import dataclasses
from types import FunctionType
from typing import Any

import numpy as np
from qiskit.quantum_info.operators import Operator, Choi
from qiskit.quantum_info.states import Statevector, DensityMatrix
from .db_fitval import FitVal


class ExperimentEncoder(json.JSONEncoder):
    """JSON Encoder for Numpy arrays and complex numbers."""

    def default(self, obj: Any) -> Any:  # pylint: disable=arguments-differ
        if isinstance(obj, np.ndarray):
            return {"__type__": "array", "__value__": obj.tolist()}
        if isinstance(obj, complex):
            return {"__type__": "complex", "__value__": [obj.real, obj.imag]}
        if dataclasses.is_dataclass(obj):
            return {
                "__type__": "__class_name__",
                "__value__": type(obj).__name__,
                "kwargs": dataclasses.asdict(obj),
            }
        if isinstance(obj, (Operator, Choi)):
            return {
                "__type__": "__class_name__",
                "__value__": type(obj).__name__,
                "args": (obj.data,),
                "kwargs": {"input_dims": obj.input_dims(), "output_dims": obj.output_dims()},
            }
        if isinstance(obj, (Statevector, DensityMatrix)):
            return {
                "__type__": "__class_name__",
                "__value__": type(obj).__name__,
                "args": (obj.data,),
                "kwargs": {"dims": obj.dims()},
            }
        if isinstance(obj, FunctionType):
            return {"__type__": "function", "__value__": obj.__name__}
        try:
            return super().default(obj)
        except TypeError:
            return {"__type__": "__class_name__", "__value__": type(obj).__name__}


class ExperimentDecoder(json.JSONDecoder):
    """JSON Decoder for Numpy arrays and complex numbers."""

    _class_init = {
        cls.__name__: cls for cls in [FitVal, Statevector, DensityMatrix, Operator, Choi]
    }

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        """Object hook."""
        if "__type__" in obj:
            if obj["__type__"] == "complex":
                val = obj["__value__"]
                return val[0] + 1j * val[1]
            if obj["__type__"] == "array":
                return np.array(obj["__value__"])
            if obj["__type__"] == "function":
                return obj["__value__"]
            if obj["__type__"] == "__class_name__":
                if obj["__value__"] in self._class_init:
                    cls = self._class_init[obj["__value__"]]
                    args = obj.get("args", tuple())
                    kwargs = obj.get("kwargs", dict())
                    return cls(*args, **kwargs)
                else:
                    return obj["__value__"]
        return obj
