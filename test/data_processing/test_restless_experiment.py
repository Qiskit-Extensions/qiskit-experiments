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

"""Test restless fine amplitude characterization and calibration experiments."""
from test.base import QiskitExperimentsTestCase

import numpy as np
from ddt import ddt, data

from qiskit.circuit import Gate

from qiskit_experiments.library import (
    FineAmplitude,
)

from qiskit_experiments.test.mock_iq_backend import MockRestlessFineAmp

from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing.nodes import Probability


@ddt
class TestFineAmpEndToEnd(QiskitExperimentsTestCase):
    """Test the fine amplitude experiment."""

    @data(0.02, 0.03, 0.04)
    def test_end_to_end_restless(self, pi_ratio):
        """Test the restless experiment end to end."""

        amp_exp = FineAmplitude(0, Gate("x", 1, []))
        amp_exp.set_transpile_options(basis_gates=["xp", "x", "sx"])
        amp_exp.set_experiment_options(add_sx=True)
        amp_exp.analysis.set_options(angle_per_gate=np.pi, phase_offset=np.pi / 2)
        amp_exp.set_run_options(rep_delay=1e-6, meas_level=2, memory=True, init_qubits=False)

        error = -np.pi * pi_ratio
        backend = MockRestlessFineAmp(error, np.pi, "x")

        expdata = amp_exp.run(backend)
        result = expdata.analysis_results(1)
        d_theta = result.value.value

        tol = 0.04

        self.assertAlmostEqual(d_theta, error, delta=tol)
        self.assertEqual(result.quality, "good")

    @data(0.02, 0.03, 0.04)
    def test_end_to_end_restless_standard_processor(self, pi_ratio):
        """Test the restless experiment with a standard processor end to end."""

        amp_exp = FineAmplitude(0, Gate("x", 1, []))
        amp_exp.set_transpile_options(basis_gates=["xp", "x", "sx"])
        amp_exp.set_experiment_options(add_sx=True)
        # standard data processor.
        standard_processor = DataProcessor("counts", [Probability("1")])
        amp_exp.analysis.set_options(
            angle_per_gate=np.pi, phase_offset=np.pi / 2, data_processor=standard_processor
        )
        amp_exp.set_run_options(rep_delay=1e-6, meas_level=2, memory=True, init_qubits=False)

        error = -np.pi * pi_ratio
        backend = MockRestlessFineAmp(error, np.pi, "x")

        expdata = amp_exp.run(backend)
        result = expdata.analysis_results(1)
        d_theta = result.value.value

        self.assertTrue(d_theta != error)
        self.assertEqual(result.quality, "bad")
