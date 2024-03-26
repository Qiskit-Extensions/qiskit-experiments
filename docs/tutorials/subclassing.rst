==================================
Subclassing an Existing Experiment
==================================

This document will take you step-by-step through the process of subclassing an existing experiment in the Qiskit Experiment module.
The example in this guide focuses on adjusting the FineAmplitude experiment to calibrate on higher order transitions.
However, a similar process can be followed for other experiments.

The FineAmplitude Experiment
============================

The ``FineAmplitude`` calibration experiment repeats N times per gate with a pulse to amplify the under-/over-rotations in the gate to determine the optimal amplitude.
This experiment can be performed for a variety of rotations and subclasses are provided for the :math:`\pi` and :math:`\frac{\pi}{2}` rotations as ``FineXAmplitude`` and ``FineSXAmplitude`` respectively.
These provided subclasses focus on the 0 <-> 1 transition, however this experiment can also be performed for higher order transitions.

Subclassing the Experiment
==========================

Our objective is to create a new class, ``HigherOrderFineXAmplitude``, which calibrates schedules on transitions other than the 0 <-> 1 transition for the :math:`\pi` rotation.
In order to do this, we need to create a subclass, shown below.

.. code-block::
   
   class HigherOrderFineXAmplitude(FineXAmplitude):
	def _pre_circuit(self) -> QuantumCircuit:
	    """Return a preparation circuit.
    
	    This method can be overridden by subclasses e.g. to calibrate schedules on
	    transitions other than the 0 <-> 1 transition.
	    """
	    circuit = QuantumCircuit(1)

	    circuit.x(0)

	    if self.experiment_options.add_sx:
	        circuit.sx(0)

	    if self.experiment_options.sx_schedule is not None:
	        sx_schedule = self.experiment_options.sx_schedule
	        circuit.add_calibration("sx", (self.physical_qubits[0],), sx_schedule, params=[])
	        circuit.barrier()

	    return circuit

In this subclass we have overridden the ``_pre_circuit`` method in order to calibrate on higher energy transitions by using an initial X gate to populate the first excited state.

Using the Subclass
==================

Now, we can use our new subclass as we would the original parent class.
Pictured below are the results from following the Fine amplitude calibration tutorial for detecting an over-rotated pulse using our new ``HigherOrderFineXAmplitude`` class in place of the original ``FineXAmplitude`` class.
You can try this for yourself and verify that your results are similar.

.. code-block::
   
   DbAnalysisResultV1
   - name: d_theta
   - value: -0.020710672666906425 ± 0.0012903658449026907
   - χ²: 0.7819653845899581
   - quality: good
   - device_components: ['Q0']
   - verified: False
