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

"""Class to test utility functions for calibrations."""

from test.base import QiskitExperimentsTestCase
import qiskit.pulse as pulse
from qiskit_experiments.calibration_management.calibration_utils import (
    used_in_calls,
    has_calls,
    get_names_called_by_name,
)
from qiskit_experiments.calibration_management.called_schedule_by_name import CalledScheduleByName
from qiskit_experiments.calibration_management.basis_gate_library import EchoedCrossResonance


class TestCalibrationUtils(QiskitExperimentsTestCase):
    """Test the function in CalUtils."""

    def test_used_in_calls(self):
        """Test that we can identify schedules by name when calls are present."""

        with pulse.build(name="xp2") as xp2:
            pulse.play(pulse.Gaussian(160, 0.5, 40), pulse.DriveChannel(1))

        xp_call = pulse.ScheduleBlock(name="call_xp")
        xp_call.append(CalledScheduleByName("xp", pulse.DriveChannel(1)))

        self.assertSetEqual(used_in_calls("xp", [xp_call]), {"call_xp"})
        self.assertSetEqual(used_in_calls("xp", [xp2]), set())

        # Check that the x gate is called by the ecr gate.
        ecr_lib = EchoedCrossResonance()

        self.assertSetEqual(used_in_calls("x", [ecr_lib["ecr"]]), {"ecr"})

    def test_has_calls(self):
        """Method to test if a schedule has calls."""

        with pulse.build(name="xp") as xp:
            pulse.play(pulse.Gaussian(160, 0.5, 40), pulse.DriveChannel(1))

        with pulse.build(name="call_xp") as call_xp:
            pulse.call(xp)

        with pulse.build(name="call_xp2") as call_xp2:
            with pulse.align_sequential():
                pulse.play(pulse.Gaussian(160, 0.5, 40), pulse.DriveChannel(1))
                pulse.call(xp)

        self.assertFalse(has_calls(xp))
        self.assertTrue(has_calls(call_xp))
        self.assertTrue(has_calls(call_xp2))
        self.assertFalse(has_calls(EchoedCrossResonance()["ecr"]))

    def test_get_names_called_by_name(self):
        """Test the get names called by name method."""

        ecr = EchoedCrossResonance()

        self.assertSetEqual(get_names_called_by_name(ecr["ecr"]), {"x"})
        self.assertSetEqual(get_names_called_by_name(ecr["cr45p"]), set())
