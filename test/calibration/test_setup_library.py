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

"""Class to test the calibrations setup methods."""

from typing import Dict, Set
import json
from test.base import QiskitExperimentsTestCase
from ddt import ddt, data, unpack

import qiskit.pulse as pulse
from qiskit.pulse.transforms import block_to_schedule
from qiskit.test.mock import FakeBelem

from qiskit_experiments.calibration_management.basis_gate_library import (
    FixedFrequencyTransmon,
    EchoedCrossResonance,
)
from qiskit_experiments.calibration_management.calibration_key_types import DefaultCalValue
from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.calibration_management.calibration_utils import compare_schedule_blocks
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.framework.json import ExperimentEncoder, ExperimentDecoder


class TestLibrary(FixedFrequencyTransmon):
    """A subclass designed for test_hash_warn.

    This class ensures that FixedFrequencyTransmon is preserved if anything goes wrong
    with the serialization :meth:`in test_hash_warn`.
    """

    def _build_schedules(self, basis_gates: Set[str]) -> Dict[str, pulse.ScheduleBlock]:
        """Dummy schedule building."""
        with pulse.build(name="x") as schedule:
            pulse.play(pulse.Drag(160, 0.1, 40, 0), pulse.DriveChannel(0))

        schedules = dict()
        if "x" in basis_gates:
            schedules["x"] = schedule

        return schedules


def _test_library_equivalence(lib1, lib2) -> bool:
    """Test if libraries are equivalent.

    Two libraries are equivalent if they have the same basis gates and
    if the strings of the schedules are equal. We cannot directly compare
    the schedules because the parameter objects in them will be different
    instances.
    """

    if len(set(lib1.basis_gates)) != len(set(lib2.basis_gates)):
        return False

    for gate in lib1.basis_gates:
        if not compare_schedule_blocks(lib1[gate], lib2[gate]):
            return False

    return True


class TestFixedFrequencyTransmon(QiskitExperimentsTestCase):
    """Test the various setup methods."""

    def test_standard_single_qubit_gates(self):
        """Test the setup of single-qubit gates."""

        library = FixedFrequencyTransmon(default_values={"duration": 320})

        for gate in ["x", "sx"]:
            sched = library[gate]
            self.assertTrue(isinstance(sched, pulse.ScheduleBlock))
            self.assertEqual(len(sched.parameters), 5)

        sched_x = library["x"]
        sched_y = library["y"]
        sched_sx = library["sx"]
        sched_sy = library["sy"]

        self.assertEqual(sched_x.blocks[0].pulse.duration, sched_sx.blocks[0].pulse.duration)
        self.assertEqual(sched_x.blocks[0].pulse.sigma, sched_sx.blocks[0].pulse.sigma)

        self.assertEqual(len(sched_x.parameters & sched_y.parameters), 4)
        self.assertEqual(len(sched_sx.parameters & sched_sy.parameters), 4)

        expected = [
            DefaultCalValue(0.5, "amp", (), "x"),
            DefaultCalValue(0.0, "β", (), "x"),
            DefaultCalValue(320, "duration", (), "x"),
            DefaultCalValue(80, "σ", (), "x"),
            DefaultCalValue(320, "duration", (), "sx"),
            DefaultCalValue(0.0, "β", (), "sx"),
            DefaultCalValue(0.25, "amp", (), "sx"),
            DefaultCalValue(80, "σ", (), "sx"),
        ]

        for param_conf in library.default_values():
            self.assertTrue(param_conf in expected)

        # Check that an error gets raise if the gate is not in the library.
        with self.assertRaises(CalibrationError):
            print(library["bswap"])

        # Test the basis gates of the library.
        self.assertListEqual(library.basis_gates, ["x", "y", "sx", "sy"])

    def test_unlinked_parameters(self):
        """Test the we get schedules with unlinked parameters."""

        library = FixedFrequencyTransmon(link_parameters=False)

        sched_x = library["x"]
        sched_y = library["y"]
        sched_sx = library["sx"]
        sched_sy = library["sy"]

        # Test the number of parameters.
        self.assertEqual(len(sched_x.parameters & sched_y.parameters), 2)
        self.assertEqual(len(sched_sx.parameters & sched_sy.parameters), 2)

        expected = [
            DefaultCalValue(0.5, "amp", (), "x"),
            DefaultCalValue(0.0, "β", (), "x"),
            DefaultCalValue(160, "duration", (), "x"),
            DefaultCalValue(40, "σ", (), "x"),
            DefaultCalValue(160, "duration", (), "sx"),
            DefaultCalValue(0.0, "β", (), "sx"),
            DefaultCalValue(0.25, "amp", (), "sx"),
            DefaultCalValue(40, "σ", (), "sx"),
            DefaultCalValue(0.5j, "amp", (), "y"),
            DefaultCalValue(0.0, "β", (), "y"),
            DefaultCalValue(160, "duration", (), "y"),
            DefaultCalValue(40, "σ", (), "y"),
            DefaultCalValue(160, "duration", (), "sy"),
            DefaultCalValue(0.0, "β", (), "sy"),
            DefaultCalValue(0.25j, "amp", (), "sy"),
            DefaultCalValue(40, "σ", (), "sy"),
        ]

        self.assertSetEqual(set(library.default_values()), set(expected))

    def test_setup_partial_gates(self):
        """Check that we do not setup all gates if not required."""

        library = FixedFrequencyTransmon(basis_gates=["x", "sy"])

        self.assertTrue("x" in library)
        self.assertTrue("sy" in library)
        self.assertTrue("y" not in library)
        self.assertTrue("sx" not in library)

        with self.assertRaises(CalibrationError):
            FixedFrequencyTransmon(basis_gates=["x", "bswap"])

    def test_serialization(self):
        """Test the serialization of the object."""

        lib1 = FixedFrequencyTransmon(
            basis_gates=["x", "sy"],
            default_values={"duration": 320},
            link_parameters=False,
        )

        lib2 = FixedFrequencyTransmon.from_config(lib1.config())

        self.assertEqual(lib2.basis_gates, lib1.basis_gates)

        # Note: we convert to string since the parameters prevent a direct comparison.
        self.assertTrue(_test_library_equivalence(lib1, lib2))

        # Test that the extra args are properly accounted for.
        lib3 = FixedFrequencyTransmon(
            basis_gates=["x", "sy"],
            default_values={"duration": 320},
            link_parameters=True,
        )

        self.assertFalse(_test_library_equivalence(lib1, lib3))

    def test_json_serialization(self):
        """Test that the library can be serialized using JSon."""

        lib1 = FixedFrequencyTransmon(
            basis_gates=["x", "sy"],
            default_values={"duration": 320},
            link_parameters=False,
        )

        # Test that serialization fails without the right encoder
        with self.assertRaises(TypeError):
            json.dumps(lib1)

        # Test that serialization works with the proper library
        lib_data = json.dumps(lib1, cls=ExperimentEncoder)
        lib2 = json.loads(lib_data, cls=ExperimentDecoder)

        self.assertTrue(_test_library_equivalence(lib1, lib2))

    def test_hash_warn(self):
        """Test that a warning is raised when the hash of the library is different.

        This test mimics the behaviour of the following workflow:
        1. A user serializes a library.
        2. Changes to the class of the library are made.
        3. The user deserializes the library with the changed class.
        4. A warning is raised since the class definition has changed.
        """

        lib1 = TestLibrary()
        lib_data = json.dumps(lib1, cls=ExperimentEncoder)
        lib2 = json.loads(lib_data, cls=ExperimentDecoder)

        self.assertTrue(_test_library_equivalence(lib1, lib2))

        # stash method build schedules to avoid other tests from failing
        build_schedules = TestLibrary._build_schedules

        def _my_build_schedules():
            """A dummy function to change the class behaviour."""
            pass

        # Change the schedule behaviour
        TestLibrary._build_schedules = _my_build_schedules

        with self.assertWarns(UserWarning):
            json.loads(lib_data, cls=ExperimentDecoder)

        TestLibrary._build_schedules = build_schedules


@ddt
class TestEchoedCrossResonance(QiskitExperimentsTestCase):
    """Test the cross-resonance extension of the fixed frequency library."""

    def test_library(self):
        """Basic test of the library."""
        self.assertTrue("ecr" in EchoedCrossResonance())

    @data(
        (1, 2, 2),
        (3, 4, 6),
    )
    @unpack
    def test_cals_initialization(self, control, target, uchan):
        """Test the an instance of Calibrations can be initialized with this library."""
        sq_lib = FixedFrequencyTransmon()
        libraries = [sq_lib, EchoedCrossResonance()]
        cals = Calibrations.from_backend(FakeBelem(), libraries)

        cr_sched = cals.get_schedule("ecr", (control, target))

        with pulse.build(name="ecr") as expected:
            with pulse.align_sequential():
                with pulse.align_left():
                    pulse.play(
                        pulse.GaussianSquare(640, 0.5, 64, 384, name="cr90p"),
                        pulse.ControlChannel(uchan),
                    )
                    pulse.play(
                        pulse.GaussianSquare(640, 0.0, 64, 384, name="rot90p"),
                        pulse.DriveChannel(target),
                    )
                pulse.play(pulse.Drag(160, 0.5, 40, 0), pulse.DriveChannel(control))
                with pulse.align_left():
                    pulse.play(
                        pulse.GaussianSquare(640, -0.5, 64, 384, name="cr90m"),
                        pulse.ControlChannel(uchan),
                    )
                    pulse.play(
                        pulse.GaussianSquare(640, 0.0, 64, 384, name="rot90m"),
                        pulse.DriveChannel(target),
                    )
                pulse.play(pulse.Drag(160, 0.5, 40, 0), pulse.DriveChannel(control))

        self.assertEqual(block_to_schedule(cr_sched), block_to_schedule(expected))

    def test_json_serialization(self):
        """Test that the library can be serialized using JSon."""

        lib1 = EchoedCrossResonance()

        # Test that serialization fails without the right encoder
        with self.assertRaises(TypeError):
            json.dumps(lib1)

        # Test that serialization works with the proper library
        lib_data = json.dumps(lib1, cls=ExperimentEncoder)
        lib2 = json.loads(lib_data, cls=ExperimentDecoder)

        self.assertTrue(_test_library_equivalence(lib1, lib2))
