# This code is part of Qiskit.
#
# (C) Copyright IBM 2019-2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Calibration helper functions"""

from typing import Optional, Set, Tuple
from functools import lru_cache
import re
import retworkx as rx

from qiskit.circuit import ParameterExpression, Parameter
from qiskit.pulse import ScheduleBlock

from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.calibration_management.calibration_key_types import ScheduleKey


# The channel indices need to be parameterized following this regex.
CHANNEL_PATTERN = r"^ch\d[\.\d]*\${0,1}[\d]*$"
CHANNEL_PATTERN_REGEX = re.compile(CHANNEL_PATTERN)


def update_schedule_dependency(schedule: ScheduleBlock, dag: rx.PyDiGraph, key: ScheduleKey):
    """Update a DAG of schedule dependencies.

    Args:
        schedule: A ScheduleBlock that potentially has references to other schedules
            that are already present in the dag.
        dag: A directed acyclic graph that encodes schedule dependencies using references.
        key: The schedule key which also contains the qubits.
    """
    parent_idx = dag.add_node(repr(key))
    for reference in schedule.references:
        ref_key = ScheduleKey(reference[0], key.qubits)
        dag.add_edge(parent_idx, _get_node_index(ref_key, dag), None)


def used_in_references(keys: Set[ScheduleKey], dag: rx.PyDiGraph) -> Set[str]:
    """Find all the schedules in the DAG that reference the given schedules.

    Args:
        keys: A list of schedules keys to which references may exist.
        dag: The dag that represents the dependencies between schedule references.

    Returns:
        A set of schedules that reference the given schedules.
    """
    callers = set()

    for key in keys:
        callers.update(_referred_by(key, dag))

    return set(key.schedule for key in callers)


def _referred_by(key: ScheduleKey, dag: rx.PyDiGraph) -> Set[ScheduleKey]:
    """Return all the schedules that refer to this schedule by name."""
    referred_by = set()

    for predecessor in dag.predecessors(_get_node_index(key, dag)):
        new_key = ScheduleKey.from_repr(predecessor)
        referred_by.add(new_key)
        referred_by.update(_referred_by(new_key, dag))

    return referred_by


def _get_node_index(key: ScheduleKey, dag: rx.PyDiGraph) -> int:
    """A helper method to get the node index in the DAG.

    If the given ScheduleKey is not found then we try and get the default schedule with the
    same name. I.e. a key for which the qubits are an empty tuple.

    Args:
        key: The ScheduleKey for which to find a node in the DAG.
        dag: The DAG of schedule dependencies.

    Returns:
        The index of the node in the dag corresponding to schedule key or its default.
    """
    try:
        return dag.nodes().index(repr(key))
    except ValueError:
        default_key = ScheduleKey(key.schedule, tuple())
        return dag.nodes().index(repr(default_key))


def validate_channels(schedule: ScheduleBlock) -> Set[Parameter]:
    """Validate and get the parameters in the channels of the schedule.

    Channels implicitly defined in references are ignored.

    Args:
        schedule: The schedule for which to get the parameters in the channels.

    Returns:
        The set of parameters explicitly defined in the schedule.

    Raises:
        CalibrationError: If a channel is parameterized by more than one parameter.
        CalibrationError: If the parameterized channel index is not formatted properly.
    """
    param_indices = set()

    # Schedules with references do not explicitly have channels. This needs special handling.
    if schedule.is_referenced():
        for block in schedule.blocks:
            if isinstance(block, ScheduleBlock):
                param_indices.update(validate_channels(block))

        return param_indices

    for ch in schedule.channels:
        if isinstance(ch.index, ParameterExpression):
            if len(ch.index.parameters) != 1:
                raise CalibrationError(f"Channel {ch} can only have one parameter.")

            param_indices.add(ch.index)
            if CHANNEL_PATTERN_REGEX.match(ch.index.name) is None:
                raise CalibrationError(
                    f"Parameterized channel must correspond to {CHANNEL_PATTERN}"
                )

    return param_indices


@lru_cache
def reference_info(
    reference: Tuple[str, ...],
    qubits: Optional[Tuple[int, ...]] = None,
) -> Tuple[str, Tuple[int, ...]]:
    """Extract reference information from the reference tuple.

    Args:
        reference: The reference of a Reference instruction in a ScheduleBlock.
        qubits: Optional argument to reorder the references.

    Returns:
        A string corresponding to the name of the referenced schedule and the qubits that
        this schedule applies to.

    Raises:
        CalibrationError: If ``reference`` is not a tuple.
        CalibrationError: If ``reference`` is not a tuple of reference name and the qubits that
            that the schedule applies to.
    """
    if not isinstance(reference, tuple):
        raise CalibrationError(f"A schedule reference must be a tuple. Found {reference}.")

    ref_schedule_name, ref_qubits = reference[0], reference[1:]

    if not isinstance(ref_schedule_name, str) and not isinstance(ref_qubits, tuple):
        raise CalibrationError(
            f"A schedule reference is a name and qubits tuple. Found {reference}"
        )

    ref_qubits = tuple(int(qubit[1:]) for qubit in ref_qubits)

    # get the qubit indices for which we are getting the schedules
    if qubits is not None and len(qubits) >= len(ref_qubits):
        ref_qubits = tuple(qubits[idx] for idx in ref_qubits)

    return ref_schedule_name, ref_qubits
