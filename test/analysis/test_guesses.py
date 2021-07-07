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

"""Test parameter guess functions."""
# pylint: disable=invalid-name

import numpy as np
from ddt import ddt, data, unpack
from qiskit.test import QiskitTestCase

from qiskit_experiments.analysis import guesses


@ddt
class TestGuesses(QiskitTestCase):
    """Test for initial guess functions."""

    __tolerance_percent__ = 0.2

    def assertAlmostEqualAbsolute(self, value: float, ref_value: float):
        """A helper validation function that assumes relative error tolerance."""
        delta = TestGuesses.__tolerance_percent__ * np.abs(ref_value)
        self.assertAlmostEqual(value, ref_value, delta=delta)

    @data(1.1, 2.0, 1.6, -1.4, 4.5)
    def test_frequency(self, freq: float):
        """Test for frequency guess."""
        x = np.linspace(-1, 1, 101)
        y = 0.3 * np.cos(2 * np.pi * freq * x + 0.5) + 1.2

        freq_guess = guesses.frequency(x, y)

        self.assertAlmostEqualAbsolute(freq_guess, np.abs(freq))

    @data(
        [0.20928722, -0.40958197, 0.29898025, 0.45622079, -0.33379813],
        [-0.41245894, -0.42868717, -0.17165843, -0.28708211, -0.25228829],
        [0.01775771, 0.47539627, 0.1101062, 0.38296899, -0.22005228],
    )
    def test_max(self, test_values):
        """Test max value."""
        max_guess, idx = guesses.max_height(test_values)
        ref_val = max(test_values)
        ref_idx = np.argmax(test_values)
        self.assertEqual(max_guess, ref_val)
        self.assertEqual(idx, ref_idx)

        max_guess, idx = guesses.max_height(test_values, absolute=True)
        ref_val = max(np.absolute(test_values))
        ref_idx = np.argmax(np.absolute(test_values))
        self.assertEqual(max_guess, ref_val)
        self.assertEqual(idx, ref_idx)

        max_guess, idx = guesses.max_height(test_values, percentile=80)
        ref_val = np.percentile(test_values, 80)
        ref_idx = np.argmin(np.abs(test_values - ref_val))
        self.assertEqual(max_guess, ref_val)
        self.assertEqual(idx, ref_idx)

    @data(
        [0.20928722, -0.40958197, 0.29898025, 0.45622079, -0.33379813],
        [-0.41245894, -0.42868717, -0.17165843, -0.28708211, -0.25228829],
        [0.01775771, 0.47539627, 0.1101062, 0.38296899, -0.22005228],
    )
    def test_min(self, test_values):
        """Test min value."""
        min_guess, idx = guesses.min_height(test_values)
        ref_val = min(test_values)
        ref_idx = np.argmin(test_values)
        self.assertEqual(min_guess, ref_val)
        self.assertEqual(idx, ref_idx)

        min_guess, idx = guesses.min_height(test_values, absolute=True)
        ref_val = min(np.absolute(test_values))
        ref_idx = np.argmin(np.absolute(test_values))
        self.assertEqual(min_guess, ref_val)
        self.assertEqual(idx, ref_idx)

        min_guess, idx = guesses.min_height(test_values, percentile=20)
        ref_val = np.percentile(test_values, 20)
        ref_idx = np.argmin(np.abs(test_values - ref_val))
        self.assertEqual(min_guess, ref_val)
        self.assertEqual(idx, ref_idx)

    @data(1.2, -0.6, 0.1, 3.5, -4.1, 3.0)
    def test_exp_decay(self, alpha: float):
        """Test for exponential decay guess."""
        x = np.linspace(0, 1, 100)
        y = np.exp(alpha * x)

        alpha_guess = guesses.exp_decay(x, y)

        self.assertAlmostEqualAbsolute(alpha_guess, alpha)

    @data([1.2, 1.4], [-0.6, 2.5], [0.1, 2.3], [3.5, 1.1], [-4.1, 6.5], [3.0, 1.2])
    @unpack
    def test_exp_osci_decay(self, alpha, freq):
        """Test of exponential decay guess with oscillation."""
        x = np.linspace(0, 1, 100)
        y = np.exp(alpha * x) * np.cos(2 * np.pi * freq * x)

        alpha_guess = guesses.oscillation_exp_decay(x, y)

        self.assertAlmostEqualAbsolute(alpha_guess, alpha)

    @data(
        [10, 1.0, 0.5],
        [50, 1.2, 0.2],
        [80, -1.2, 0.6],
        [30, -0.2, 0.4],
        [40, 3.2, 0.3],
        [20, -0.4, 0.8],
    )
    @unpack
    def test_linewidth_spect(self, idx, a, fwhm):
        """Test of linewidth of peaks."""
        x = np.linspace(-1, 1, 100)
        sigma = fwhm / np.sqrt(8 * np.log(2))
        y = a * np.exp(-((x - x[idx]) ** 2) / (2 * sigma ** 2))

        lw_guess = guesses.full_width_half_max(x, y, idx)

        self.assertAlmostEqual(fwhm, lw_guess, delta=0.1)

    @data(
        [0.1, 0.0, 1.0, 0.5],
        [-0.3, 0.6, 1.2, 0.2],
        [0.2, -0.8, -1.2, 0.6],
        [0.9, 0.2, -0.2, 0.4],
        [0.6, 0.1, 3.2, 0.3],
        [-0.7, -0.4, -1.6, 0.8],
    )
    @unpack
    def test_baseline_spect(self, b0, x0, a, fwhm):
        """Test of baseline of peaks."""
        x = np.linspace(-1, 1, 100)
        sigma = fwhm / np.sqrt(8 * np.log(2))
        y = a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + b0

        b0_guess = guesses.constant_spectral_offset(y)

        self.assertAlmostEqual(b0, b0_guess, delta=0.1)

    @data(
        [0.1, 0.0, 1.0, 1.3],
        [-0.3, 0.6, 1.2, 0.4],
        [0.2, -0.8, -1.2, 3.6],
        [0.9, 0.2, -0.2, 0.3],
        [0.6, 0.1, 3.2, 0.8],
        [-0.7, -0.4, -1.6, 1.2],
    )
    @unpack
    def test_baseline_sinusoidal(self, b0, x0, a, freq):
        """Test of baseline of sinusoidal signal."""
        x = np.linspace(-1, 1, 100)
        y = a * np.cos(2 * np.pi * freq * (x - x0)) + b0

        b0_guess = guesses.constant_sinusoidal_offset(y)

        self.assertAlmostEqual(b0, b0_guess, delta=0.1)
