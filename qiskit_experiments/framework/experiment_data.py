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
Experiment Data class
"""
from __future__ import annotations
import logging
from typing import Dict, Optional, List, Union, Any, TYPE_CHECKING
from datetime import datetime
import warnings
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit_experiments.database_service import DbExperimentDataV1 as DbExperimentData
from qiskit_experiments.database_service.database_service import (
    DatabaseServiceV1 as DatabaseService,
    DeviceComponent,
)
from qiskit_experiments.database_service.utils import ThreadSafeOrderedDict

if TYPE_CHECKING:
    # There is a cyclical dependency here, but the name needs to exist for
    # Sphinx on Python 3.9+ to link type hints correctly.  The gating on
    # `TYPE_CHECKING` means that the import will never be resolved by an actual
    # interpreter, only static analysis.
    from . import BaseExperiment

LOG = logging.getLogger(__name__)


class ExperimentData(DbExperimentData):
    """Qiskit Experiments Data container class"""

    def __init__(
        self,
        experiment: Optional["BaseExperiment"] = None,
        backend: Optional[Backend] = None,
        parent_id: Optional[str] = None,
        job_ids: Optional[List[str]] = None,
        child_data: Optional[List[ExperimentData]] = None,
    ):
        """Initialize experiment data.

        Args:
            experiment: Optional, experiment object that generated the data.
            backend: Optional, Backend the experiment runs on.
            parent_id: Optional, ID of the parent experiment data
                in the setting of a composite experiment
            job_ids: Optional, IDs of jobs submitted for the experiment.
            child_data: Optional, list of child experiment data.
        """
        if experiment is not None:
            backend = backend or experiment.backend
            experiment_type = experiment.experiment_type
        else:
            experiment_type = None

        self._experiment = experiment
        super().__init__(
            experiment_type=experiment_type,
            backend=backend,
            parent_id=parent_id,
            job_ids=job_ids,
            metadata=experiment._metadata() if experiment else {},
        )

        # Add component data and set parent ID to current container
        self._child_data = ThreadSafeOrderedDict()
        if child_data is not None:
            self._set_child_data(child_data)

    @property
    def experiment(self):
        """Return the experiment for this data.

        Returns:
            BaseExperiment: the experiment object.
        """
        return self._experiment

    @property
    def completion_times(self) -> Dict[str, datetime]:
        """Returns the completion times of the jobs."""
        job_times = {}
        for job_id, job in self._jobs.items():
            if job is not None and "COMPLETED" in job.time_per_step():
                job_times[job_id] = job.time_per_step().get("COMPLETED")

        return job_times

    def add_child_data(self, experiment_data: ExperimentData):
        """Add child experiment data to the current experiment data"""
        experiment_data._parent_id = self.experiment_id
        self._child_data[experiment_data.experiment_id] = experiment_data

    def child_data(
        self, index: Optional[Union[int, slice, str]] = None
    ) -> Union[ExperimentData, List[ExperimentData]]:
        """Return child experiment data.

        Args:
            index: Index of the child experiment data to be returned.
                Several types are accepted for convenience:

                    * None: Return all child data.
                    * int: Specific index of the child data.
                    * slice: A list slice of indexes.
                    * str: experiment ID of the child data.

        Returns:
            The requested single or list of child experiment data.

        Raises:
            QiskitError: if the index or ID of the child experiment data
                         cannot be found.
        """
        if index is None:
            return self._child_data.values()
        if isinstance(index, (int, slice)):
            return self._child_data.values()[index]
        if isinstance(index, str):
            return self._child_data[index]
        raise QiskitError(f"Invalid index type {type(index)}.")

    def component_experiment_data(
        self, index: Optional[Union[int, slice]] = None
    ) -> Union[ExperimentData, List[ExperimentData]]:
        """Return child experiment data"""
        warnings.warn(
            "This method is deprecated and will be removed next release. "
            "Use the `child_data` method instead.",
            DeprecationWarning,
        )
        return self.child_data(index)

    def data(
        self,
        index: Optional[Union[int, slice, str]] = None,
    ) -> List[CircuitData]:
        """Return the experiment data at the specified index.

        Args:
            index: Index of the data to be returned.
                Several types are accepted for convenience:

                    * None: Return all experiment data.
                    * int: Specific index of the data.
                    * slice: A list slice of data indexes.
                    * str: ID of the job that produced the data.

        Returns:
            Experiment data.

        Raises:
            TypeError: If the input `index` has an invalid type.
        """
        raw_data = super().data(index=index)

        if isinstance(raw_data, dict):
            raw_data = [raw_data]
        return [CircuitData.from_dict(raw_datum) for raw_datum in raw_data]

    def save(self) -> None:
        super().save()
        for data in self._child_data.values():
            original_verbose = data.verbose
            data.verbose = False
            data.save()
            data.verbose = original_verbose

    def save_metadata(self) -> None:
        super().save_metadata()
        for data in self.child_data():
            data.save_metadata()

    def _save_experiment_metadata(self):
        # Copy child experiment IDs to metadata
        if self._child_data:
            self._metadata["child_data_ids"] = self._child_data.keys()
        super()._save_experiment_metadata()

    @classmethod
    def load(cls, experiment_id: str, service: DatabaseService) -> ExperimentData:
        expdata = DbExperimentData.load(experiment_id, service)
        expdata.__class__ = ExperimentData
        expdata._experiment = None
        child_data_ids = expdata.metadata.pop("child_data_ids", [])
        child_data = [ExperimentData.load(child_id, service) for child_id in child_data_ids]
        expdata._set_child_data(child_data)
        return expdata

    def copy(self, copy_results=True) -> "ExperimentData":
        new_instance = super().copy(copy_results=copy_results)

        # Copy additional attributes not in base class
        if self.experiment is None:
            new_instance._experiment = None
        else:
            new_instance._experiment = self.experiment.copy()

        # Recursively copy child data
        child_data = [data.copy(copy_results=copy_results) for data in self.child_data()]
        new_instance._set_child_data(child_data)
        return new_instance

    def _set_child_data(self, child_data: List[ExperimentData]):
        """Set child experiment data for the current experiment."""
        self._child_data = ThreadSafeOrderedDict()
        for data in child_data:
            self.add_child_data(data)

    def _set_service(self, service: DatabaseService) -> None:
        """Set the service to be used for storing experiment data,
           to this experiment itself and its descendants.

        Args:
            service: Service to be used.

        Raises:
            DbExperimentDataError: If an experiment service is already being used.
        """
        super()._set_service(service)
        for data in self.child_data():
            data._set_service(service)

    @DbExperimentData.share_level.setter
    def share_level(self, new_level: str) -> None:
        """Set the experiment share level,
           to this experiment itself and its descendants.

        Args:
            new_level: New experiment share level. Valid share levels are provider-
                specified. For example, IBM Quantum experiment service allows
                "public", "hub", "group", "project", and "private".
        """
        self._share_level = new_level
        for data in self._child_data.values():
            original_auto_save = data.auto_save
            data.auto_save = False
            data.share_level = new_level
            data.auto_save = original_auto_save
        if self.auto_save:
            self.save_metadata()

    def add_tags_recursive(self, tags2add: List[str]) -> None:
        """Add tags to this experiment itself and its descendants

        Args:
            tags2add - the tags that will be added to the existing tags
        """
        self.tags += tags2add
        for data in self._child_data.values():
            data.add_tags_recursive(tags2add)

    def remove_tags_recursive(self, tags2remove: List[str]) -> None:
        """Remove tags from this experiment itself and its descendants

        Args:
            tags2remove - the tags that will be removed from the existing tags
        """
        self.tags = [x for x in self.tags if x not in tags2remove]
        for data in self._child_data.values():
            data.remove_tags_recursive(tags2remove)

    def __repr__(self):
        out = (
            f"<ExperimentData[{self.experiment_type}]"
            f", backend: {self.backend}"
            f", status: {self.status()}"
            f", experiment_id: {self.experiment_id}>"
        )
        return out

    def __str__(self):
        line = 51 * "-"
        n_res = len(self._analysis_results)
        status = self.status()
        ret = line
        ret += f"\nExperiment: {self.experiment_type}"
        ret += f"\nExperiment ID: {self.experiment_id}"
        if self._parent_id:
            ret += f"\nParent ID: {self._parent_id}"
        if self._child_data:
            ret += f"\nChild Experiment Data: {len(self._child_data)}"
        ret += f"\nStatus: {status}"
        if status == "ERROR":
            ret += "\n  "
            ret += "\n  ".join(self._errors)
        if self.backend:
            ret += f"\nBackend: {self.backend}"
        if self.tags:
            ret += f"\nTags: {self.tags}"
        ret += f"\nData: {len(self._data)}"
        ret += f"\nAnalysis Results: {n_res}"
        ret += f"\nFigures: {len(self._figures)}"
        return ret


class CircuitData:
    """Outcome with metadata from the single quantum circuit execution."""

    __slots__ = (
        "_counts",
        "_memory",
        "_job_id",
        "_metadata",
        "_shots",
        "_meas_level",
        "_meas_return",
    )

    def __init__(
        self,
        job_id,
        metadata: Dict[str, Any],
        counts: Optional[Dict[str, Union[int, float]]] = None,
        memory: Optional[np.ndarray] = None,
        shots: Optional[int] = None,
        meas_level: Optional[int] = None,
        meas_return: Optional[str] = None,
    ):
        """Create new data object.

        Args:
            job_id: ID of Qiskit job generating this data.
            metadata: Metadata associated with this circuit.
            counts: Optional, counts data dictionary.
            memory: Optional, memory data array-like.
            shots: Optional, number of repeated measurement.
            meas_level: Optional, measurement level.
            meas_return: Optional, data format of returned data, averaged or not.
        """
        self._job_id = job_id
        self._metadata = metadata
        self._counts = counts
        self._memory = np.asarray(memory)
        self._shots = shots
        self._meas_level = meas_level
        self._meas_return = meas_return

    @property
    def job_id(self):
        """Return job id. Read-only."""
        return self._job_id

    @property
    def metadata(self):
        """Return memory data. Read-only."""
        return self._metadata

    @property
    def shots(self):
        """Return shot number. Read-only.

        Raises:
            ValueError: When value is not set.
        """
        if self._shots is None:
            raise ValueError("Shot information is not available.")
        return self._shots

    @property
    def meas_level(self) -> MeasLevel:
        """Return measurement level.

        Raises:
            ValueError: When value is not set.
        """
        if self._meas_level is None:
            raise ValueError("Measurement level is not available.")
        return MeasLevel(self._meas_level)

    @meas_level.setter
    def meas_level(self, meas_level: MeasLevel):
        """Set new measurement level after data processing.

        Args:
            meas_level: New measurement level.
        """
        self.meas_level = meas_level.value

    @property
    def meas_return(self) -> MeasReturnType:
        """Return measurement data format.

        Raises:
            ValueError: When value is not set.
        """
        if self._meas_return is None:
            raise ValueError("Meas return type is not available.")
        return MeasReturnType(self._meas_return)

    @meas_return.setter
    def meas_return(self, meas_return: MeasReturnType):
        """Set new measurement data format after data processing.

        Args:
            meas_return: New measurement data format.
        """
        self._meas_return = meas_return.value

    @property
    def counts(self):
        """Return count data.

        Raises:
            ValueError: When value is not set.
        """
        if self._counts is None:
            raise ValueError("Count data is not available.")
        return self._counts

    @counts.setter
    def counts(self, counts: Dict[str, Union[int, float]]):
        """Set count data after data processing.

        Args:
            counts: Count value to set.
        """
        self._counts = counts

    @property
    def memory(self) -> np.ndarray:
        """Return memory data.

        Raises:
            ValueError: When value is not set.
        """
        if self._memory is None:
            raise ValueError("Memory data is not available.")
        return self._memory

    @memory.setter
    def memory(self, memory: np.ndarray):
        """Set memory data array after data processing.

        Args:
            memory: Memory data array to set.
        """
        self._memory = np.asarray(memory)

    def filter(self, **kwargs) -> bool:
        """Check if this is target circuit outcome.

        Returns:
            Return ``True`` when all input key-value pair matches with this metadata.
        """
        return all(self._metadata.get(k, None) == v for k, v in kwargs.items())

    def get(self, key: str, default_value=None) -> Any:
        """Return specific attribute. Backward compatibility.

        Args:
            key: Name of attribute to return.
            default_value: Value returned when the entry doesn't exist.

        Returns:
            Value of the attribute.
        """
        try:
            return getattr(self, key)
        except (ValueError, AttributeError):
            return default_value

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "CircuitData":
        """Initialize instance from dictionary that database service generates.

        Args:
            data_dict: Free form dictionary of circuit result.

        Returns:
            CircuitData instance.
        """
        return cls(
            job_id=data_dict["job_id"],
            metadata=data_dict["metadata"],
            counts=data_dict.get("counts", None),
            memory=data_dict.get("memory", None),
            shots=data_dict.get("shots", None),
            meas_level=data_dict.get("meas_level", None),
            meas_return=data_dict.get("meas_return", None),
        )

    def __getitem__(self, item):
        # for backward compatibility
        return getattr(self, item)

    def __repr__(self):
        # this returns only metadata, which is the most important
        # identifier of data entry in Qiskit Experiments
        configs = ", ".join(f"{k}={v}" for k, v in self.metadata.items())
        return f"{self.__class__.__name__}({configs})"
