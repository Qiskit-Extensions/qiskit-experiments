===================================
Subclassing the CurveAnalysis Class
===================================

This document will take you step-by-step through the process of subclassing the CurveAnalysis class in the Qiskit Experiment module.
The example in this guide focuses on creating a new analysis class for the Ramsey XY experiment.
However, a similar process can be followed to create analysis classes for other experiments.

Creating the Class
==================

When analyzing the Ramsey XY experiment, our objective is to fit the X and Y series to a cosine and sine function, respectively.
Both functions share the same frequency and amplitude parameters.
First, we need to define our new class:

.. code-block::
   
   class RamseyXYAnalysis(curve.CurveAnalysis):

Within this new class we need to define the `__series__` attribute.
This attribute defines the set of points to be fit in the fit function.
Since we are fitting both a cosine and a sine function, we need to define two `SeriesDef` elements:

.. code-block::
   
       __series__ = [
        curve.SeriesDef(
            fit_func=lambda x, amp, tau, freq, base, phase: fit_function.cos_decay(
                x, amp=amp, tau=tau, freq=freq, phase=phase, baseline=base
            ),
            plot_color="blue",
            name="X",
            filter_kwargs={"series": "X"},
            plot_symbol="o",
            model_description=r"{\rm amp} e^{-x/\tau} \cos\left(2 \pi\cdot {\rm freq}\cdot x "
            r"+ {\rm phase}) + {\rm base}",
        ),
        curve.SeriesDef(
            fit_func=lambda x, amp, tau, freq, base, phase: fit_function.sin_decay(
                x, amp=amp, tau=tau, freq=freq, phase=phase, baseline=base
            ),
            plot_color="green",
            name="Y",
            filter_kwargs={"series": "Y"},
            plot_symbol="^",
            model_description=r"{\rm amp} e^{-x/\tau} \sin\left(2 \pi\cdot {\rm freq}\cdot x "
            r"+ {\rm phase}\right) + {\rm base}",
        ),
    ]

Each `SeriesDef` object defines a custom `fit_func` to which the data will be fit along with several other parameters defining how the data will be presented.
Our `fit_func` methods make use of the provided `cos_decay` and `sin_decay` methods in CurveAnalysis.

Additional Options
==================

The `__series__` attribute is the only attribute required to be overriden in the new class, however there are additional methods such as `_default_options`, `_evaluate_quality`, and others which you may want to override depending on your experiment.
More information about all of these can be found in the CurveAnalysis source file.
