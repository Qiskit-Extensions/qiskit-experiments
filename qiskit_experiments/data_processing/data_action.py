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

"""Defines the steps that can be used to analyse data."""

from abc import ABCMeta, abstractmethod
from typing import Generator, Iterator, Optional

import numpy as np


class DataAction(metaclass=ABCMeta):
    """
    Abstract action done on measured data to process it. Each subclass of DataAction must
    define the way it formats, validates and processes data.
    """

    def __init__(self, validate: bool = True):
        """
        Args:
            validate: If set to False the DataAction will not validate its input.
        """
        self._validate = validate

    def _process(self, gen_datum: Iterator) -> Generator:
        """Applies the data processing step to the datum.

        Args:
            gen_datum: A generator of unprocessed data. Each entry is a tuple of data and error.

        Yields:
            A tuple of processed data and error.
        """
        yield from gen_datum

    def _format_data(self, gen_datum: Iterator) -> Generator:
        """Validate and format the input.

        Check that the given data and error have the correct structure.

        Args:
            gen_datum: A generator of unformatted data. Each entry is a tuple of data and error.

        Yields:
            A tuple of formatted data and error.
        """
        yield from gen_datum

    def __call__(self, gen_datum: Iterator) -> Generator:
        """Call the data action of this node on the data and propagate the error.

        Args:
            gen_datum: A generator of raw data. Each entry is a tuple of data and error.

        Yields:
            A generator that implements a data processing pipeline.
        """
        yield from self._process(self._format_data(gen_datum))

    def __repr__(self):
        """String representation of the node."""
        return f"{self.__class__.__name__}(validate={self._validate})"


class TrainableDataAction(DataAction):
    """A base class for data actions that need training."""

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Return False if the DataAction needs to be trained.

        Subclasses must implement this property to communicate if they have been trained.

        Return:
            True if the data action has been trained.
        """

    @abstractmethod
    def train(self, full_val_arr: np.ndarray, full_err_arr: Optional[np.ndarray] = None):
        """Train a DataAction.

        Certain data processing nodes, such as a SVD, require data to first train.

        Args:
            full_val_arr: A list of values. Each datum will be converted to a 2D array.
            full_err_arr: A list of errors. Each datm will be converted to a 2D array.
        """
