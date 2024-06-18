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

from qiskit_experiments.library import (
    FineXAmplitude,
)

from qiskit_experiments.test.mock_iq_backend import MockRestlessFineAmp

from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing.nodes import Probability
from qiskit_experiments.framework import Options
from qiskit_experiments.data_processing.exceptions import DataProcessorError


@ddt
class TestFineAmpEndToEndRestless(QiskitExperimentsTestCase):
    """Test the fine amplitude experiment in a restless measurement setting."""

    def test_enable_restless(self):
        """Test the enable_restless method."""

        error = -np.pi * 0.01
        backend = MockRestlessFineAmp(error, np.pi, "x")
        amp_exp = FineXAmplitude([0], backend)

        with self.assertRaises(DataProcessorError):
            amp_exp.enable_restless(rep_delay=2.0)
            amp_exp.run()

        amp_exp = FineXAmplitude([0], backend)
        amp_exp.enable_restless(rep_delay=2.0, suppress_t1_error=True)

        self.assertTrue(
            amp_exp.run_options,
            Options(
                meas_level=2, rep_delay=2.0, init_qubits=False, memory=True, use_measure_esp=False
            ),
        )

    @data(-0.03, -0.01, 0.02, 0.04)
    def test_end_to_end_restless(self, pi_ratio):
        """Test the restless experiment end to end."""

        error = -np.pi * pi_ratio
        backend = MockRestlessFineAmp(error, np.pi, "x")

        amp_exp = FineXAmplitude([0], backend)
        amp_exp.enable_restless(rep_delay=1e-6)

        expdata = amp_exp.run(backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results("d_theta")
        d_theta = result.value.n

        self.assertAlmostEqual(d_theta, error, delta=0.01)
        self.assertEqual(result.quality, "good")

        # check that the fit amplitude is almost 1 as expected.
        amp_fit = expdata.artifacts("fit_summary").data.params["amp"]
        self.assertAlmostEqual(amp_fit, 1.0, delta=0.02)

    @data(-0.02, 0.04)
    def test_end_to_end_restless_standard_processor(self, pi_ratio):
        """Test the restless experiment with a standard processor end to end."""

        error = -np.pi * pi_ratio
        backend = MockRestlessFineAmp(error, np.pi, "x")

        amp_exp = FineXAmplitude([0], backend)
        # standard data processor.
        standard_processor = DataProcessor("counts", [Probability("01")])
        amp_exp.analysis.set_options(data_processor=standard_processor)
        # enable a restless measurement setting.
        amp_exp.enable_restless(rep_delay=1e-6, override_processor_by_restless=False)

        expdata = amp_exp.run(backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results("d_theta")
        d_theta = result.value.n

        self.assertTrue(abs(d_theta - error) > 0.01)

        # check that the fit amplitude is much smaller than 1.
        amp_fit = expdata.artifacts("fit_summary").data.params["amp"]
        self.assertTrue(amp_fit < 0.05)
