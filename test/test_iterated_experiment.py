# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Classes to test the IteratedExperiment class."""

import numpy as np

from qiskit.test import QiskitTestCase

from qiskit_experiments.framework import ExperimentData, Options
from qiskit_experiments.framework.iterated_experiment import IteratedExperiment
from qiskit_experiments.test.mock_iq_backend import MockFineAmp
from qiskit_experiments.library.calibration.fine_amplitude import FineXAmplitudeCal
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.calibration_management import Calibrations


class TestIteratedExperiment(QiskitTestCase):
    """Test the iterated experiment class.

    Meaningful tests require calibration experiments.
    """

    def setUp(self):
        """Setup the tests"""
        super().setUp()

        library = FixedFrequencyTransmon()

        error = 0.01 * np.pi
        self.backend = MockFineAmp(error, np.pi, "x")
        self.cals = Calibrations.from_backend(self.backend, library)

    def test_iterated_fine_amp(self):
        """Test that we can iterate a fine amplitude experiment."""

        def callback_for_test(exp_data: ExperimentData, options: Options):
            """Callback to modify the error of the backend to fake a real setting."""
            error = options.backend.angle_error
            options.backend.angle_error = error / 2

            if exp_data.analysis_results(-1).value.value < options.tol:
                return False

            return True

        # Before the experiment runs, the amp parameter should have two values.
        self.assertTrue(len(self.cals.parameters_table(parameters=["amp"])["data"]) == 2)

        amp_cal = FineXAmplitudeCal(0, self.cals, "x", backend=self.backend)
        iter_exp = IteratedExperiment(amp_cal, callback=callback_for_test)

        # Add the backend to pass it to the callback through the options to fake a real setting.
        iter_exp.set_run_options(backend=self.backend, tol=0.005)

        # Run the experiment. With the chosen settings there should be more than one child exp_data.
        exp_data = iter_exp.run()

        self.assertTrue(len(exp_data.child_data()) > 4)
        self.assertTrue(len(exp_data.child_data()) < 10)  # max number of iterations.

        # The last result should be below the tolerance and all other ones above.
        tol = iter_exp.run_options.tol
        n_iter = len(exp_data.child_data())
        for idx, child_data in enumerate(exp_data.child_data()):
            if idx < n_iter - 1:
                self.assertTrue(child_data.analysis_results(-1).value.value > tol)
            else:
                self.assertTrue(child_data.analysis_results(-1).value.value < tol)

        # After the experiment runs, the amp parameter should have more than two values.
        amp_params = self.cals.parameters_table(parameters=["amp"], most_recent_only=False)["data"]
        self.assertTrue(len(amp_params) == (2 + n_iter))
