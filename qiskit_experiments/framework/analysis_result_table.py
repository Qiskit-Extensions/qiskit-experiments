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

"""A table-like dataset for analysis results."""
from __future__ import annotations

import re
import threading
import uuid
import warnings
from typing import Any

import numpy as np
import pandas as pd

from qiskit_experiments.database_service.exceptions import ExperimentEntryNotFound


class AnalysisResultTable:
    """A table-like dataset for analysis results.

    Default table columns are defined in the class attribute :attr:`.DEFAULT_COLUMNS`.
    The table is automatically expanded when an extra key is included in the
    input dictionary data. Missing columns in the input data are filled with a null value.

    Table row index (i.e. entry ID) is created by truncating the result_id string which
    is basically a UUID-4 string. A random unique ID is generated when the result_id
    is missing in the input data.

    Any operation on the table value via the instance methods guarantees thread safety.
    """

    VALID_ID_REGEX = re.compile(r"\A(?P<short_id>\w{8})-\w{4}-\w{4}-\w{4}-\w{12}\Z")

    DEFAULT_COLUMNS = [
        "name",
        "experiment",
        "components",
        "value",
        "quality",
        "experiment_id",
        "result_id",
        "tags",
        "backend",
        "run_time",
        "created_time",
    ]

    def __init__(self):
        """Create new dataset."""
        self._data = pd.DataFrame(columns=self.DEFAULT_COLUMNS)
        self._lock = threading.RLock()

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame) -> "AnalysisResultTable":
        """Create new dataset with existing dataframe.

        Args:
            data: Bare dataframe object.

        Returns:
            A new AnalysisResults instance.
        """
        instance = AnalysisResultTable()
        instance._data = pd.concat([instance._data, data])
        return instance

    @property
    def dataframe(self) -> pd.DataFrame:
        """Dataframe object of analysis results."""
        with self._lock:
            return self._data.copy(deep=False)

    @property
    def result_ids(self) -> list[str]:
        """Result IDs in current dataset."""
        with self._lock:
            return list(self._data.result_id)

    @property
    def columns(self) -> list[str]:
        """All columns in current dataset."""
        with self._lock:
            return list(self._data.columns)

    def add_data(
        self,
        *,
        result_id: str | None = None,
        **data,
    ) -> str:
        """Add new data to this dataset.

        Args:
            result_id: A unique UUID-4 string for this data entry.
                The full string is used to identify the data in the experiment service database,
                and a short ID is created by truncating this string as a dataframe index.
            data: Arbitrary key-value pairs representing a single data entry.
                Missing values for default columns are filled with ``None``.

        Returns:
            Assigned analysis result ID.
        """
        result_id = result_id or self._create_unique_hash()

        if matched := re.match(self.VALID_ID_REGEX, result_id):
            # Short unique index is generated from result id.
            # Showing full result id unnecessary occupies horizontal space of the html table.
            # This mechanism is inspired by the github commit hash.
            index = matched.group("short_id")
        else:
            warnings.warn(
                f"Result ID of {result_id} is not a valid UUID-4 string. ",
                UserWarning,
            )
            index = result_id[:8]

        with self._lock:
            if index in self._data.index:
                raise ValueError(
                    f"Table entry index {index} already exists. "
                    "Please use another ID to avoid index collision."
                )

            # Add missing columns to the table
            if missing := data.keys() - set(self._data.columns):
                for k in data:
                    # Order sensitive
                    if k in missing:
                        loc = len(self._data.columns)
                        self._data.insert(loc, k, value=None)

            # A hack to avoid unwanted dtype update. Appending new row with .loc indexer
            # performs enlargement and implicitly changes dtype. This often induces a confusion of
            # NaN (numeric container) and None (object container) for missing values.
            # Filling a row with None values before assigning actual values can keep column dtype,
            # but this behavior might change in future pandas version.
            # https://github.com/pandas-dev/pandas/issues/6485
            # Also see test.framework.test_data_table.TestBaseTable.test_type_*
            self._data.loc[index, :] = [None] * len(self._data.columns)
            template = dict.fromkeys(self.columns, None)
            template["result_id"] = result_id
            template.update(data)
            self._data.loc[index, :] = pd.array(list(template.values()), dtype=object)

        return index

    def get_data(
        self,
        key: str | int | slice | None = None,
        columns: str | list[str] = "default",
    ) -> pd.DataFrame:
        """Get matched entries from this dataset.

        Args:
            key: Identifier of the entry of interest.
            columns: List of names or a policy (default, minimal, all)
                of data columns included in the returned data frame.

        Returns:
            Matched entries in a single data frame or series.
        """
        if key is None:
            with self._lock:
                out = self._data.copy()
        else:
            uids = self._resolve_key(key)
            with self._lock:
                out = self._data.filter(items=uids, axis=0)
        if columns != "all":
            valid_columns = self._resolve_columns(columns)
            out = out[valid_columns]
        return out

    def del_data(
        self,
        key: str | int,
    ) -> list[str]:
        """Delete matched entries from this dataset.

        Args:
            key: Identifier of the entry of interest.

        Returns:
            Deleted analysis result IDs.
        """
        uids = self._resolve_key(key)
        with self._lock:
            self._data.drop(uids, inplace=True)

        return uids

    def clear(self):
        """Clear all table entries."""
        with self._lock:
            self._data = pd.DataFrame(columns=self.DEFAULT_COLUMNS)

    def copy(self, new_ids: bool = True):
        """Create new thread-safe instance with the same data.

        Args:
            new_ids: Whether to generate new IDs for copied entries. Defaults to True.

        Returns:
            A new shallow copied DataFrame object.
        """
        with self._lock:
            # Hold the lock so that no data can be added
            new_instance = self.__class__()
            new_instance._data = self._data.copy(deep=False)
            if new_ids:
                new_instance._data["result_id"] = None
                for idx, _ in new_instance._data.iterrows():
                    new_instance._data.at[idx, "result_id"] = new_instance._create_unique_hash()
                new_instance._data.index = [
                    result_id[:8] for result_id in new_instance._data["result_id"]
                ]
        return new_instance

    def _create_unique_hash(self) -> str:
        with self._lock:
            n = 0
            while n < 1000:
                tmp_id = str(uuid.uuid4())
                if tmp_id[:8] not in self._data.index:
                    return tmp_id
        raise RuntimeError(
            "Unique result_id string cannot be prepared for this table within 1000 trials. "
            "Reduce number of entries, or manually provide a unique result_id."
        )

    def _resolve_columns(self, columns: str | list[str]):
        with self._lock:
            extra_columns = [c for c in self._data.columns if c not in self.DEFAULT_COLUMNS]
            if columns == "default":
                return [
                    "name",
                    "experiment",
                    "components",
                    "value",
                    "quality",
                    "backend",
                    "run_time",
                ] + extra_columns
            if columns == "minimal":
                return [
                    "name",
                    "components",
                    "value",
                    "quality",
                ] + extra_columns
            if not isinstance(columns, str):
                out = []
                for column in columns:
                    if column in self._data.columns:
                        out.append(column)
                    else:
                        warnings.warn(
                            f"Specified column {column} does not exist in this table.",
                            UserWarning,
                        )
                return out
        raise ValueError(
            f"Column group {columns} is not valid name. Use either 'all', 'default', 'minimal'."
        )

    def _resolve_key(self, key: int | slice | str) -> list[str]:
        with self._lock:
            if isinstance(key, int):
                if key >= len(self):
                    raise ExperimentEntryNotFound(f"Analysis result {key} not found.")
                return [self._data.index[key]]
            if isinstance(key, slice):
                keys = list(self._data.index)[key]
                if len(keys) == 0:
                    raise ExperimentEntryNotFound(f"Analysis result {key} not found.")
                return keys
            if isinstance(key, str):
                if key in self._data.index:
                    return [key]
                # This key is name of entry
                loc = self._data["name"] == key
                if not any(loc):
                    raise ExperimentEntryNotFound(f"Analysis result {key} not found.")
                return list(self._data.index[loc])

        raise TypeError(f"Invalid key type {type(key)}. The key must be either int, slice, or str.")

    def __len__(self):
        return len(self._data)

    def __contains__(self, item):
        return item in self._data.index

    def __json_encode__(self) -> dict[str, Any]:
        with self._lock:
            return {
                "class": "AnalysisResultTable",
                "data": self._data.to_dict(orient="index"),
            }

    @classmethod
    def __json_decode__(cls, value: dict[str, Any]) -> "AnalysisResultTable":
        if not value.get("class", None) == "AnalysisResultTable":
            raise ValueError("JSON decoded value for AnalysisResultTable is not valid class type.")

        instance = object.__new__(cls)
        instance._lock = threading.RLock()
        instance._data = pd.DataFrame.from_dict(
            data=value.get("data", {}),
            orient="index",
        ).replace({np.nan: None})
        return instance

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.RLock()
