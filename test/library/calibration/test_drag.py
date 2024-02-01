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

"""Test drag calibration experiment."""

from test.base import QiskitExperimentsTestCase
import unittest
from ddt import ddt, data, unpack
import numpy as np

from qiskit import pulse
from qiskit.circuit import Parameter
from qiskit.exceptions import QiskitError
from qiskit.providers.fake_provider import FakeWashingtonV2
from qiskit.pulse import DriveChannel, Drag
from qiskit.qobj.utils import MeasLevel

from qiskit_experiments.library import RoughDrag, RoughDragCal
from qiskit_experiments.library.characterization.analysis import DragCalAnalysis
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.test import SingleTransmonTestBackend


@ddt
class TestDragEndToEnd(QiskitExperimentsTestCase):
    """Test the drag experiment."""

    def setUp(self):
        """Setup some schedules."""
        super().setUp()

        beta = Parameter("β")
        self.backend = SingleTransmonTestBackend(noise=False, atol=1e-3)
        with pulse.build(name="xp") as xp:
            pulse.play(
                Drag(duration=160, amp=0.5 / self.backend.rabi_rate_01[0], sigma=40, beta=beta),
                DriveChannel(0),
            )

        self.x_plus = xp
        self.test_tol = 0.1

    @data(
        (None, None),
        (np.linspace(-4, 4, 31), {"beta": 1.5, "freq": 0.08}),
    )
    @unpack
    def test_end_to_end(self, betas, p0_opt):
        """Test the drag experiment end to end."""

        drag = RoughDrag([0], self.x_plus)
        drag.set_run_options(shots=200)

        if betas is not None:
            drag.set_experiment_options(betas=betas)
        if p0_opt:
            drag.analysis.set_options(p0=p0_opt)

        expdata = drag.run(self.backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results(1)

        # pylint: disable=no-member
        self.assertTrue(abs(result.value.n - self.backend.beta_01[0]) < self.test_tol)
        self.assertEqual(result.quality, "good")
        self.assertEqual(expdata.metadata["meas_level"], MeasLevel.CLASSIFIED)

    @data(
        ([1, 3, 5], None, 0.2),  # partial oscillation.
        ([3, 5, 7], None, 0.2),  # constant offset, i.e. lower SNR.
        ([1, 3, 5], np.linspace(-1, 1, 51), 0.2),  # Beta not in range
        ([1, 3, 5], np.linspace(-2.5, 2.5, 51), 0.2),  # Max closer to zero
    )
    @unpack
    def test_nasty_data(self, reps, betas, tol):
        """A set of tests for non-ideal data."""

        drag = RoughDrag([0], self.x_plus, betas=betas)
        drag.set_experiment_options(reps=reps)
        drag.set_run_options(shots=500)

        exp_data = drag.run(self.backend)
        self.assertExperimentDone(exp_data)
        result = exp_data.analysis_results("beta")
        # pylint: disable=no-member
        self.assertTrue(abs(result.value.n - self.backend.beta_01[0]) < tol)
        self.assertEqual(result.quality, "good")

    def test_drag_reanalysis(self):
        """Test edge case for re-analyzing DRAG result multiple times."""

        drag = RoughDrag([0], self.x_plus)
        drag.set_experiment_options(reps=[2, 4, 6])

        expdata = drag.run(self.backend, analysis=None)
        self.assertExperimentDone(expdata)

        # Assume the situation we reloaded experiment data from server
        # and prepared analysis class in the client machine.
        # DRAG reps numbers might be different from the default value,
        # but the client doesn't know the original setting.
        analysis = DragCalAnalysis()
        expdata1 = analysis.run(expdata.copy(), replace_results=True)
        self.assertExperimentDone(expdata1)
        # Check mapping of model name to circuit metadata.
        self.assertDictEqual(
            analysis.options.data_subfit_map,
            {
                "nrep=2": {"nrep": 2},
                "nrep=4": {"nrep": 4},
                "nrep=6": {"nrep": 6},
            },
        )

        # Running experiment twice.
        # Reported by https://github.com/Qiskit/qiskit-experiments/issues/1086.
        expdata2 = analysis.run(expdata.copy(), replace_results=True)
        self.assertExperimentDone(expdata2)
        self.assertEqual(len(analysis.models), 3)

        self.assertAlmostEqual(
            expdata1.analysis_results("beta").value.n,
            expdata2.analysis_results("beta").value.n,
        )


class TestDragCircuits(QiskitExperimentsTestCase):
    """Test the circuits of the drag calibration."""

    def setUp(self):
        """Setup some schedules."""
        super().setUp()

        beta = Parameter("β")

        with pulse.build(name="xp") as xp:
            pulse.play(Drag(duration=160, amp=0.208519, sigma=40, beta=beta), DriveChannel(0))

        self.x_plus = xp

    def test_default_circuits(self):
        """Test the default circuit."""

        drag = RoughDrag([0], self.x_plus)
        drag.set_experiment_options(reps=[2, 4, 8])
        circuits = drag._transpiled_circuits()

        for idx, expected in enumerate([4, 8, 16]):
            ops = circuits[idx * 51].count_ops()
            self.assertEqual(ops["Drag(xp)"], expected)

    def test_circuit_roundtrip_serializable(self):
        """Test circuit serializations for drag experiment."""
        drag = RoughDrag([0], self.x_plus)
        drag.set_experiment_options(reps=[2, 4], betas=[-5, 5])
        drag.backend = FakeWashingtonV2()
        self.assertRoundTripSerializable(drag._transpiled_circuits())

    def test_raise_multiple_parameter(self):
        """Check that the experiment raises with unassigned parameters."""

        beta = Parameter("β")
        amp = Parameter("amp")

        with pulse.build(name="xp") as xp:
            pulse.play(Drag(duration=160, amp=amp, sigma=40, beta=beta), DriveChannel(0))

        with self.assertRaises(QiskitError):
            RoughDrag([1], xp, betas=np.linspace(-3, 3, 21))


class TestRoughDragCalUpdate(QiskitExperimentsTestCase):
    """Test that a Drag calibration experiment properly updates the calibrations."""

    def setUp(self):
        """Setup the tests"""
        super().setUp()

        library = FixedFrequencyTransmon()

        self.backend = SingleTransmonTestBackend(noise=False, atol=1e-3)
        self.cals = Calibrations.from_backend(self.backend, libraries=[library])
        self.test_tol = 0.05

    # pylint: disable=no-member
    def test_update(self):
        """Test that running RoughDragCal updates the calibrations."""

        qubit = 0
        prev_beta = self.cals.get_parameter_value("β", (0,), "x")
        self.assertEqual(prev_beta, 0)

        expdata = RoughDragCal([qubit], self.cals, backend=self.backend).run()
        self.assertExperimentDone(expdata)

        new_beta = self.cals.get_parameter_value("β", (0,), "x")
        self.assertTrue(abs(new_beta) > self.test_tol)

    def test_dragcal_experiment_config(self):
        """Test RoughDragCal config can round trip"""
        exp = RoughDragCal([0], self.cals, backend=self.backend)
        loaded_exp = RoughDragCal.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqualExtended(exp, loaded_exp)

    @unittest.skip("Calibration experiments are not yet JSON serializable")
    def test_dragcal_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = RoughDragCal([0], self.cals)
        self.assertRoundTripSerializable(exp)

    def test_drag_experiment_config(self):
        """Test RoughDrag config can roundtrip"""
        with pulse.build(name="xp") as sched:
            pulse.play(pulse.Drag(160, 0.5, 40, Parameter("β")), pulse.DriveChannel(0))
        exp = RoughDrag([0], backend=self.backend, schedule=sched)
        loaded_exp = RoughDrag.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqualExtended(exp, loaded_exp)

    @unittest.skip("Schedules are not yet JSON serializable")
    def test_drag_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        with pulse.build(name="xp") as sched:
            pulse.play(pulse.Drag(160, 0.5, 40, Parameter("β")), pulse.DriveChannel(0))
        exp = RoughDrag([0], backend=self.backend, schedule=sched)
        self.assertRoundTripSerializable(exp)
