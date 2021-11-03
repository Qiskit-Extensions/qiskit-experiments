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

"""Actions done on the data to bring it in a usable form."""

import itertools
from typing import Any, Dict, List, Set, Tuple, Union, Generator, Iterator

import numpy as np

from qiskit_experiments.data_processing.data_action import DataAction, TrainableDataAction
from qiskit_experiments.data_processing.exceptions import DataProcessorError


class DataProcessor:
    """
    A DataProcessor defines a sequence of operations to perform on experimental data.
    Calling an instance of DataProcessor applies this sequence on the input argument.
    A DataProcessor is created with a list of DataAction instances. Each DataAction
    applies its _process method on the data and returns the processed data. The nodes
    in the DataProcessor may also perform data validation and some minor formatting.
    The output of one data action serves as input for the next data action.
    DataProcessor.__call__(datum) usually takes in an entry from the data property of
    an ExperimentData object (i.e. a dict containing metadata and memory keys and
    possibly counts, like the Result.data property) and produces the formatted data.
    DataProcessor.__call__(datum) extracts the data from the given datum under
    DataProcessor._input_key (which is specified at initialization) of the given datum.
    """

    def __init__(
        self,
        input_key: str,
        data_actions: Union[DataAction, TrainableDataAction] = None,
    ):
        """Create a chain of data processing actions.

        Args:
            input_key: The initial key in the datum Dict[str, Any] under which the data processor
                will find the data to process.
            data_actions: A list of data processing actions to construct this data processor with.
                If None is given an empty DataProcessor will be created.
        """
        self._input_key = input_key
        self._nodes = data_actions if data_actions else []

    def append(self, node: Union[DataAction, TrainableDataAction]):
        """
        Append new data action node to this data processor.

        Args:
            node: A DataAction that will process the data.
        """
        self._nodes.append(node)

    @property
    def is_trained(self) -> bool:
        """Return True if all nodes of the data processor have been trained."""
        for node in self._nodes:
            if isinstance(node, TrainableDataAction):
                if not node.is_trained:
                    return False

        return True

    def __call__(self, data: Union[Dict, List[Dict]], **options) -> Tuple[Any, Any]:
        """
        Call self on the given datum. This method sequentially calls the stored data actions
        on the datum.

        Args:
            data: The data, typically from ExperimentData.data(...), that needs to be processed.
            This dict or list of dicts also contains the metadata of each experiment.
            options: Run-time options given as keyword arguments that will be passed to the nodes.

        Returns:
            processed data: The data processed by the data processor.
        """
        return self._call_internal(data, **options)

    def call_with_history(
        self, data: Union[Dict, List[Dict]], history_nodes: Set = None
    ) -> Tuple[Any, Any, List]:
        """
        Call self on the given datum. This method sequentially calls the stored data actions
        on the datum and also returns the history of the processed data.

        Args:
            data: The data, typically from ExperimentData.data(...), that needs to be processed.
            This dict or list of dicts also contains the metadata of each experiment.
            history_nodes: The nodes, specified by index in the data processing chain, to
                include in the history. If None is given then all nodes will be included
                in the history.

        Returns:
            processed data: The datum processed by the data processor.
            history: The datum processed at each node of the data processor.
        """
        return self._call_internal(data, True, history_nodes)

    def _call_internal(
        self,
        data: Union[Dict, List[Dict]],
        with_history: bool = False,
        history_nodes: Set = None,
        call_up_to_node: int = None,
    ) -> Union[Tuple[Any, Any], Tuple[Any, Any, List]]:
        """Process the data with or without storing the history of the computation.

        Args:
            data: The data, typically from ExperimentData.data(...), that needs to be processed.
            This dict or list of dicts also contains the metadata of each experiment.
            with_history: if True the history is returned otherwise it is not.
            history_nodes: The nodes, specified by index in the data processing chain, to
                include in the history. If None is given then all nodes will be included
                in the history.
            call_up_to_node: The data processor will use each node in the processing chain
                up to the node indexed by call_up_to_node. If this variable is not specified
                then all nodes in the data processing chain will be called.

        Returns:
            When ``with_history`` is ``False`` it returns a tuple of array-like of data and error.
            Otherwise it returns a tuple of above with a list of intermediate data at each step.
        """
        if call_up_to_node is None:
            call_up_to_node = len(self._nodes)

        # This is generator
        gen_datum = self._data_extraction(data)

        history = []
        for index, node in enumerate(self._nodes[:call_up_to_node]):
            # Create pipeline of data processing
            gen_datum = node(gen_datum)

            if with_history and (history_nodes is None or index in history_nodes):
                # make sure not to kill pipeline by execution
                gen_datum, gen_datum_copy = itertools.tee(gen_datum)
                out_values, out_errors = execute_pipeline(gen_datum_copy)
                history.append((node.__class__.__name__, out_values, out_errors, index))

        # Execute pipeline
        out_values, out_errors = execute_pipeline(gen_datum)

        # Return only first element if length=1, e.g. [[0, 1]] -> [0, 1]
        if out_values.shape[0] == 1:
            out_values = out_values[0]

        # Return only first element if length=1, e.g. [[0, 1]] -> [0, 1]
        if out_errors.shape[0] == 1:
            out_errors = out_errors[0]

        # Return None if error is not computed
        if np.isnan(out_errors).all():
            out_errors = None

        if with_history:
            return out_values, out_errors, history
        else:
            return out_values, out_errors

    def train(self, data: List[Dict[str, Any]]):
        """Train the nodes of the data processor.

        Args:
            data: The data to use to train the data processor.
        """
        for index, node in enumerate(self._nodes):
            if isinstance(node, TrainableDataAction):
                if not node.is_trained:
                    # Process the data up to the untrained node.
                    node.train(*self._call_internal(data, call_up_to_node=index))

    def _data_extraction(self, data: Union[Dict, List[Dict]]) -> Generator:
        """Extracts the data on which to run the nodes.

        If the datum is a list of dicts then the data under self._input_key is extracted
        from each dict and appended to a list which therefore contains all the data. If the
        data processor has to_array set to True then the list will be converted to a numpy
        array.

        Args:
            data: A list of such dicts where the data is contained under the key self._input_key.

        Yields:
            A tuple of numpy array object representing a data and error.

        Raises:
            DataProcessorError:
                - If the input datum is not a list or a dict.
                - If the input key of the data processor is not contained in the data.
        """
        if isinstance(data, dict):
            data = [data]

        for datum in data:
            try:
                target = datum[self._input_key]

                # returns data and initial error
                if isinstance(target, dict):
                    # likely level2 data, forcibly convert into array
                    yield np.asarray([target], dtype=object), np.asarray([np.nan], dtype=float)
                else:
                    try:
                        # level1 or below
                        nominal_arr = np.asarray(target, dtype=float)
                        stdev_arr = np.full_like(target, np.nan, dtype=float)
                    except TypeError:
                        # level2 memory ["00", "11", "01", ...]
                        nominal_arr = np.asarray(target, dtype=object)
                        stdev_arr = np.asarray([np.nan], dtype=float)
                    yield nominal_arr, stdev_arr

            except KeyError as error:
                raise DataProcessorError(
                    f"The input key {self._input_key} was not found in the input datum."
                ) from error
            except TypeError as error:
                raise DataProcessorError(
                    f"{self.__class__.__name__} only extracts data from "
                    f"lists or dicts, received {type(data)}."
                ) from error

    def __repr__(self):
        """String representation of data processors."""
        names = ", ".join(node.__class__.__name__ for node in self._nodes)

        return f"{self.__class__.__name__}(input_key={self._input_key}, nodes=[{names}])"


def execute_pipeline(gen_datum: Iterator) -> Tuple[np.ndarray, np.ndarray]:
    """Execute processing pipeline and return processed data array.

    Args:
        gen_datum: A generator to sequentially return datum.

    Returns:
        A tuple of nominal values and standard errors.
    """
    out_values, out_errors = list(zip(*gen_datum))

    try:
        # try to convert into float object for performance
        out_values = np.asarray(out_values, dtype=float)
    except TypeError:
        # if not convert into arbitrary array
        out_values = np.asarray(out_values, dtype=object)

    # convert into 1D array e.g. [[0], [1], ...] -> [0, 1, ...]
    if len(out_values.shape) == 2 and out_values.shape[1] == 1:
        out_values = out_values[:, 0]

    try:
        # try to convert into float object for performance
        out_errors = np.asarray(out_errors, dtype=float)
    except TypeError:
        # if not convert into arbitrary array
        out_errors = np.asarray(out_errors, dtype=object)

    # convert into 1D array e.g. [[0], [1], ...] -> [0, 1, ...]
    if len(out_errors.shape) == 2 and out_errors.shape[1] == 1:
        out_errors = out_errors[:, 0]

    return out_values, out_errors
