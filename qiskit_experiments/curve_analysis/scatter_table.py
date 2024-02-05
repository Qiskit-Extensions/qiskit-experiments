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

"""Table representation of the x, y data for curve fitting."""
from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from typing import Any
from itertools import groupby
from operator import itemgetter

import numpy as np
import pandas as pd

from qiskit.utils import deprecate_func


LOG = logging.getLogger(__name__)


class ScatterTable:
    """A table-like dataset for curve fitting intermediate data.

    Default table columns are defined in the class attribute :attr:`.DEFAULT_COLUMNS`.
    This table cannot be expanded with user-provided column names.

    This dataset is not thread safe. Do not use the same instance in multiple threads.

    .. _filter_scatter_table:

    Filtering ScatterTable
    ----------------------

    ScatterTable is a single source of the truth as the data used in the curve fit analysis.
    Each data point in a 1-D curve fit may consist of the x value, y value, and
    standard error of the y value.
    In addition, such analysis may internally create several data subsets,
    and data points can also take metadata triplet (`data_uid`, `category`, `analysis`)
    to distinguish the subset.

    * The `data_uid` is an integer key representing the class of the data.
      When an analysis consists of multiple fit models and performs the multi-objective fit,
      the created table may contain multiple datasets for each fit model.
      Usually the index of data matches with the index of the fit model in the analysis.
      The table also provides `name` column which is a human-friendly text notation of the data_uid.
      The name and corresponding data_uid must refer to the identical data class,
      and the name typically matches with the name of the fit model.
      You can find a particular data subset by either data_uid or name.

    * The `category` is a string key representing a tag of data groups.
      The measured outcomes input as-is to the curve analysis are tagged with "raw".
      In a standard :class:`.CurveAnalysis` subclass, the input data is pre-processed for
      the fitting and the formatted data is also stored in the table with "formatted" tag.
      After the fit is successfully conducted and the model parameters are identified,
      data points in the interpolated fit curves are also stored with "fitted" tag
      for visualization. The management of data group depends on the design of
      the curve analysis protocol, and the convention of category naming might
      be different in a particular analysis.

    * The `analysis` is a string key representing a name of
      the analysis instance that generated the data point.
      This allows a user to combine multiple tables from the different analyses
      without collapsing the data points.
      In the :class:`.CompositeCurveAnalysis`, the instance consists of statistically
      independent fit models represented in a form of nested component analysis instances.
      Such component has unique analysis name, and datasets generated from each instance
      are merged into a single table stored in the outermost composite analysis.

    User must be aware of this triplet to extract data points that belong to a
    particular data subset. For example,

    .. code-block:: python

        mini_table = table.filter(kind="model1", category="raw", analysis="Analysis_A")
        mini_x = mini_table.x
        mini_y = mini_table.y

    this operation is equivalent to

    .. code-block:: python

        mini_x = table.xvals(kind="model1", category="raw", analysis="Analysis_A")
        mini_y = table.yvals(kind="model1", category="raw", analysis="Analysis_A")

    When an analysis only has a single model and the table is created from a single
    analysis instance, the data_uid and analysis are trivial, and you only need to
    specify the category to get subset data of interest.

    """

    DEFAULT_COLUMNS = [
        "xval",
        "yval",
        "yerr",
        "name",
        "data_uid",
        "category",
        "shots",
        "analysis",
    ]

    DEFAULT_DTYPES = [
        "Float64",
        "Float64",
        "Float64",
        "string",
        "Int64",
        "string",
        "Int64",
        "string",
    ]

    def __init__(self):
        super().__init__()
        self._lazy_add_rows = []
        self._dump = pd.DataFrame(columns=self.DEFAULT_COLUMNS)

    @property
    def _data(self) -> pd.DataFrame:
        if self._lazy_add_rows:
            # Add data when table element is called.
            # Adding rows in loop is extremely slow in pandas.
            tmp_df = pd.DataFrame(self._lazy_add_rows, columns=self.DEFAULT_COLUMNS)
            tmp_df = self._format_table(tmp_df)
            if len(self._dump) == 0:
                self._dump = tmp_df
            else:
                self._dump = pd.concat([self._dump, tmp_df], ignore_index=True)
            self._lazy_add_rows.clear()
        return self._dump

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame) -> "ScatterTable":
        """Create new dataset with existing dataframe.

        Args:
            data: Data dataframe object.

        Returns:
            A new ScatterTable instance.
        """
        if list(data.columns) != cls.DEFAULT_COLUMNS:
            raise ValueError("Input dataframe columns don't match with the ScatterTable spec.")
        instance = object.__new__(ScatterTable)
        instance._lazy_add_rows = []
        instance._dump = cls._format_table(data)
        return instance

    @property
    def dataframe(self):
        """Dataframe object of data points."""
        return self._data

    @property
    def x(self) -> np.ndarray:
        """X values."""
        # For backward compatibility with CurveData.x
        return self._data.xval.to_numpy(dtype=float, na_value=np.nan)

    @x.setter
    def x(self, new_values):
        self._data.loc[:, "xval"] = new_values

    def xvals(
        self,
        kind: int | str | None = None,
        category: str | None = None,
        analysis: str | None = None,
        check_unique: bool = True,
    ) -> np.ndarray:
        """Get subset of X values.

        A convenient shortcut of getting X data with filtering.

        Args:
            kind: Identifier of the data, either data UID or name.
            category: Name of data category.
            analysis: Name of analysis.
            check_unique: Set True to check if multiple series are contained.
                When multiple series are contained, it raises a user warning.

        Returns:
            Numpy array of X values.
        """
        sub_table = self.filter(kind, category, analysis)
        if check_unique:
            self._warn_composite_data(sub_table)
        return sub_table.x

    @property
    def y(self) -> np.ndarray:
        """Y values."""
        # For backward compatibility with CurveData.y
        return self._data.yval.to_numpy(dtype=float, na_value=np.nan)

    @y.setter
    def y(self, new_values: np.ndarray):
        self._data.loc[:, "yval"] = new_values

    def yvals(
        self,
        kind: int | str | None = None,
        category: str | None = None,
        analysis: str | None = None,
        check_unique: bool = True,
    ) -> np.ndarray:
        """Get subset of Y values.

        A convenient shortcut of getting Y data with filtering.

        Args:
            kind: Identifier of the data, either data UID or name.
            category: Name of data category.
            analysis: Name of analysis.
            check_unique: Set True to check if multiple series are contained.
                When multiple series are contained, it raises a user warning.

        Returns:
            Numpy array of Y values.
        """
        sub_table = self.filter(kind, category, analysis)
        if check_unique:
            self._warn_composite_data(sub_table)
        return sub_table.y

    @property
    def y_err(self) -> np.ndarray:
        """Standard deviation of Y values."""
        # For backward compatibility with CurveData.y_err
        return self._data.yerr.to_numpy(dtype=float, na_value=np.nan)

    @y_err.setter
    def y_err(self, new_values: np.ndarray):
        self._data.loc[:, "yerr"] = new_values

    def yerrs(
        self,
        kind: int | str | None = None,
        category: str | None = None,
        analysis: str | None = None,
        check_unique: bool = True,
    ) -> np.ndarray:
        """Get subset of standard deviation of Y values.

        A convenient shortcut of getting Y error data with filtering.

        Args:
            kind: Identifier of the data, either data UID or name.
            category: Name of data category.
            analysis: Name of analysis.
            check_unique: Set True to check if multiple series are contained.
                When multiple series are contained, it raises a user warning.

        Returns:
            Numpy array of Y error values.
        """
        sub_table = self.filter(kind, category, analysis)
        if check_unique:
            self._warn_composite_data(sub_table)
        return sub_table.y_err

    @property
    def name(self) -> np.ndarray:
        """Corresponding data name."""
        return self._data.name.to_numpy(dtype=object, na_value=None)

    @name.setter
    def name(self, new_values: np.ndarray):
        self._data.loc[:, "name"] = new_values

    @property
    def data_uid(self) -> np.ndarray:
        """Corresponding data UID."""
        return self._data.data_uid.to_numpy(dtype=object, na_value=None)

    @data_uid.setter
    def data_uid(self, new_values: np.ndarray):
        self._data.loc[:, "data_uid"] = new_values

    @property
    def category(self) -> np.ndarray:
        """Category of data points."""
        return self._data.category.to_numpy(dtype=object, na_value=None)

    @category.setter
    def category(self, new_values: np.ndarray):
        self._data.loc[:, "category"] = new_values

    @property
    def shots(self) -> np.ndarray:
        """Shot number used to acquire data points."""
        return self._data.shots.to_numpy(dtype=object, na_value=np.nan)

    @shots.setter
    def shots(self, new_values: np.ndarray):
        self._data.loc[:, "shots"] = new_values

    @property
    def analysis(self) -> np.ndarray:
        """Corresponding analysis name."""
        return self._data.analysis.to_numpy(dtype=object, na_value=None)

    @analysis.setter
    def analysis(self, new_values: np.ndarray):
        self._data.loc[:, "analysis"] = new_values

    def filter(
        self,
        kind: int | str | None = None,
        category: str | None = None,
        analysis: str | None = None,
    ) -> ScatterTable:
        """Filter data by class, category, and/or analysis name.

        Args:
            kind: Identifier of the data, either data UID or name.
            category: Name of data category.
            analysis: Name of analysis.

        Returns:
            New ScatterTable object with filtered data.
        """
        filt_data = self._data

        if kind is not None:
            if isinstance(kind, int):
                index = self._data.data_uid == kind
            elif isinstance(kind, str):
                index = self._data.name == kind
            else:
                raise ValueError(f"Invalid kind type {type(kind)}. This must be integer or string.")
            filt_data = filt_data.loc[index, :]
        if category is not None:
            index = self._data.category == category
            filt_data = filt_data.loc[index, :]
        if analysis is not None:
            index = self._data.analysis == analysis
            filt_data = filt_data.loc[index, :]
        return ScatterTable.from_dataframe(filt_data)

    def iter_by_data(self) -> Iterator[tuple[int, "ScatterTable"]]:
        """Iterate over subset of data sorted by the data UID.

        Yields:
            Tuple of data UID and subset of ScatterTable.
        """
        data_ids = self._data.data_uid.dropna().sort_values().unique()
        for did in data_ids:
            index = self._data.data_uid == did
            yield did, ScatterTable.from_dataframe(self._data.loc[index, :])

    def iter_groups(
        self,
        *group_by: str,
    ) -> Iterator[tuple[tuple[Any, ...], "ScatterTable"]]:
        """Iterate over the subset sorted by multiple column values.

        Args:
            group_by: Name of column to group by.

        Yields:
            Tuple of keys and subset of ScatterTable.
        """
        try:
            sort_by = itemgetter(*[self.DEFAULT_COLUMNS.index(c) for c in group_by])
        except ValueError as ex:
            raise ValueError(
                f"Specified columns don't exist: {group_by} are not subset of {self.DEFAULT_COLUMNS}."
            ) from ex

        # Use python native groupby method on dataframe ndarray when sorting by multiple columns.
        # This is more performant than pandas groupby implementation.
        for vals, sub_data in groupby(sorted(self._data.values, key=sort_by), key=sort_by):
            tmp_df = pd.DataFrame(list(sub_data), columns=self.DEFAULT_COLUMNS)
            yield vals, ScatterTable.from_dataframe(tmp_df)

    def add_row(
        self,
        name: str | pd.NA = pd.NA,
        data_uid: int | pd.NA = pd.NA,
        category: str | pd.NA = pd.NA,
        x: float | pd.NA = pd.NA,
        y: float | pd.NA = pd.NA,
        y_err: float | pd.NA = pd.NA,
        shots: float | pd.NA = pd.NA,
        analysis: str | pd.NA = pd.NA,
    ):
        """Add new data group to the table.

        Data must be the same length.

        Args:
            x: X value.
            y: Y value.
            y_err: Standard deviation of y value.
            shots: Shot number used to acquire this data point.
            name: Name of this data if available.
            data_uid: Data UID of if available.
            category: Data category if available.
            analysis: Analysis name if available.
        """
        self._lazy_add_rows.append([x, y, y_err, name, data_uid, category, shots, analysis])

    @classmethod
    def _format_table(cls, data: pd.DataFrame) -> pd.DataFrame:
        return (
            data.replace(np.nan, pd.NA)
            .astype(dict(zip(cls.DEFAULT_COLUMNS, cls.DEFAULT_DTYPES)))
            .reset_index(drop=True)
        )

    @staticmethod
    def _warn_composite_data(table: ScatterTable):
        if len(table._data.name.unique()) > 1:
            warnings.warn(
                "Returned data contains multiple data kinds. "
                "You may want to filter the data by a specific data_uid or name.",
                UserWarning,
            )
        if len(table._data.category.unique()) > 1:
            warnings.warn(
                "Returned data contains multiple categories. "
                "You may want to filter the data by a specific category name.",
                UserWarning,
            )
        if len(table._data.analysis.unique()) > 1:
            warnings.warn(
                "Returned data contains multiple datasets from different component analyses. "
                "You may want to filter the data by a specific analysis name.",
                UserWarning,
            )

    @property
    @deprecate_func(
        since="0.6",
        additional_msg="Curve data uses dataframe representation. Call .model_id instead.",
        pending=True,
        package_name="qiskit-experiments",
        is_property=True,
    )
    def data_allocation(self) -> np.ndarray:
        """Index of corresponding fit model."""
        return self.data_uid

    @property
    @deprecate_func(
        since="0.6",
        additional_msg="No alternative is provided. Use .name with set operation.",
        pending=True,
        package_name="qiskit-experiments",
        is_property=True,
    )
    def labels(self) -> list[str]:
        """List of model names."""
        # Order sensitive
        name_id_tups = self._data.groupby(["name", "data_uid"]).groups.keys()
        return [k[0] for k in sorted(name_id_tups, key=lambda k: k[1])]

    @deprecate_func(
        since="0.6",
        additional_msg="Use filter method instead.",
        pending=True,
        package_name="qiskit-experiments",
    )
    def get_subset_of(self, index: str | int) -> "ScatterTable":
        """Filter data by series name or index.

        Args:
            index: Series index of name.

        Returns:
            A subset of data corresponding to a particular series.
        """
        return self.filter(kind=index)

    def __len__(self):
        """Return the number of data points stored in the table."""
        return len(self._data)

    def __eq__(self, other):
        return self.dataframe.equals(other.dataframe)

    def __json_encode__(self) -> dict[str, Any]:
        return {
            "class": "ScatterTable",
            "data": self._data.to_dict(orient="index"),
        }

    @classmethod
    def __json_decode__(cls, value: dict[str, Any]) -> "ScatterTable":
        if not value.get("class", None) == "ScatterTable":
            raise ValueError("JSON decoded value for ScatterTable is not valid class type.")
        tmp_df = pd.DataFrame.from_dict(value.get("data", {}), orient="index")
        return ScatterTable.from_dataframe(tmp_df)
