Calibrations: Schedules and gate parameters from experiments 
============================================================

To produce high fidelity quantum operations, we want to be able to run good gates. The
calibration module in Qiskit Experiments allows users to run experiments to find the
pulse shapes and parameter values that maximize the fidelity of the resulting quantum
operations. Calibration experiments encapsulate the internal processes and allow
experimenters to perform calibration operations in a quicker way. Without the
experiments module, we would need to define pulse schedules and plot the resulting
measurement data manually.

In this tutorial, we demonstrate how to calibrate single-qubit gates using the
calibration framework in Qiskit Experiments. We will run experiments on our test pulse
backend, :class:`.SingleTransmonTestBackend`, a backend that simulates the underlying
pulses with :mod:`qiskit_dynamics` on a three-level model of a transmon. You can also
run these experiments on any real backend with Pulse enabled (see
:class:`qiskit.providers.models.BackendConfiguration`).

We will run experiments to 
find the qubit frequency, calibrate the amplitude of DRAG pulses, and choose the value 
of the DRAG parameter that minimizes leakage. The calibration framework requires 
the user to:

- Set up an instance of :class:`.Calibrations`,

- Run calibration experiments found in :mod:`qiskit_experiments.library.calibration`.

Note that the values of the parameters stored in the instance of the :class:`.Calibrations` class 
will automatically be updated by the calibration experiments. 
This automatic updating can also be disabled using the ``auto_update`` flag.

.. note::
    This tutorial requires the :mod:`qiskit_dynamics` package to run simulations.
    You can install it with ``python -m pip install qiskit-dynamics``.

.. jupyter-execute::

    import pandas as pd
    import numpy as np
    import qiskit.pulse as pulse
    from qiskit.circuit import Parameter
    from qiskit_experiments.calibration_management.calibrations import Calibrations
    from qiskit import schedule
    from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend

.. jupyter-execute::

    backend = SingleTransmonTestBackend(noise=False, seed=100)
    qubit = 0 
    cals=Calibrations.from_backend(backend)
    print(cals.get_inst_map())

The two functions below show how to set up an instance of :class:`.Calibrations`. 
To do this the user defines the template schedules to calibrate. 
These template schedules are fully parameterized, even the channel indices 
on which the pulses are played. Furthermore, the name of the parameter in the channel 
index must follow the convention laid out in the documentation 
of the calibration module. Note that the parameters in the channel indices 
are automatically mapped to the channel index when :meth:`.Calibrations.get_schedule` is called.

.. jupyter-execute::
    
    # A function to instantiate calibrations and add a couple of template schedules.
    def setup_cals(backend) -> Calibrations:
    
        cals = Calibrations.from_backend(backend)
        
        dur = Parameter("dur")
        amp = Parameter("amp")
        sigma = Parameter("σ")
        beta = Parameter("β")
        drive = pulse.DriveChannel(Parameter("ch0"))

        # Define and add template schedules.
        with pulse.build(name="xp") as xp:
            pulse.play(pulse.Drag(dur, amp, sigma, beta), drive)

        with pulse.build(name="xm") as xm:
            pulse.play(pulse.Drag(dur, -amp, sigma, beta), drive)

        with pulse.build(name="x90p") as x90p:
            pulse.play(pulse.Drag(dur, Parameter("amp"), sigma, Parameter("β")), drive)

        cals.add_schedule(xp, num_qubits=1)
        cals.add_schedule(xm, num_qubits=1)
        cals.add_schedule(x90p, num_qubits=1)

        return cals

    # Add guesses for the parameter values to the calibrations.
    def add_parameter_guesses(cals: Calibrations):
        
        for sched in ["xp", "x90p"]:
            cals.add_parameter_value(80, "σ", schedule=sched)
            cals.add_parameter_value(0.5, "β", schedule=sched)
            cals.add_parameter_value(320, "dur", schedule=sched)
            cals.add_parameter_value(0.5, "amp", schedule=sched)

When setting up the calibrations we add three pulses: a :math:`\pi`-rotation, 
with a schedule named ``xp``, a schedule ``xm`` identical to ``xp`` 
but with a nagative amplitude, and a :math:`\pi/2`-rotation, with a schedule 
named ``x90p``. Here, we have linked the amplitude of the ``xp`` and ``xm`` pulses. 
Therefore, calibrating the parameters of ``xp`` will also calibrate 
the parameters of ``xm``.

.. jupyter-execute::

    cals = setup_cals(backend)
    add_parameter_guesses(cals)

A similar setup is achieved by using a pre-built library of gates. 
The library of gates provides a standard set of gates and some initial guesses 
for the value of the parameters in the template schedules. 
This is shown below using the ``FixedFrequencyTransmon`` library which provides the ``x``,
``y``, ``sx``, and ``sy`` pulses. Note that in the example below 
we change the default value of the pulse duration to 320 samples

.. jupyter-execute::

    from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon

    library = FixedFrequencyTransmon(default_values={"duration": 320})
    cals = Calibrations.from_backend(backend, libraries=[library])
    print(library.default_values()) # check what parameter values this library has
    print(cals.get_inst_map()) # check the new cals's InstructionScheduleMap made from the library
    print(cals.get_schedule('x',(0,))) # check one of the schedules built from the new calibration

We are going to run the spectroscopy, Rabi, DRAG, and fine amplitude calibration experiments 
one after another and update the parameters after every experiment, keeping track of
parameter values. 

Finding qubits with spectroscopy
--------------------------------

Here, we are using a backend for which we already know the qubit frequency. 
We will therefore use the spectroscopy experiment to confirm that 
there is a resonance at the qubit frequency reported by the backend.

.. jupyter-execute::

    from qiskit_experiments.library.calibration.rough_frequency import RoughFrequencyCal

We first show the contents of the calibrations for qubit 0. 
Note that the guess values that we added before apply to all qubits on the chip. 
We see this in the table below as an empty tuple ``()`` in the qubits column. 
Observe that the parameter values of ``y`` do not appear in this table as they are given by the values of ``x``.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    # dataframe styling
    pd.set_option('display.precision', 5)
    pd.set_option('display.html.border', 1)
    pd.set_option('display.max_colwidth', 24)

.. jupyter-execute::

    columns_to_show = ["parameter", "qubits", "schedule", "value", "date_time"]
    pd.DataFrame(**cals.parameters_table(qubit_list=[qubit, ()]))[columns_to_show]

Instantiate the experiment and draw the first circuit in the sweep:

.. jupyter-execute::

    freq01_estimate = backend.qubit_properties(0).frequency
    frequencies = np.linspace(freq01_estimate-15e6, freq01_estimate+15e6, 51)
    spec = RoughFrequencyCal((qubit,), cals, frequencies, backend=backend)
    spec.set_experiment_options(amp=0.005)

.. jupyter-execute::

    circuit = spec.circuits()[0]
    circuit.draw(output="mpl", style="iqp")

We can also visualize the pulse schedule for the circuit:

.. jupyter-execute::

    next(iter(circuit.calibrations["Spec"].values())).draw()   
    circuit.calibrations["Spec"]

Run the calibration experiment:

.. jupyter-execute::

    spec_data = spec.run().block_for_results()
    spec_data.figure(0) 


.. jupyter-execute::

    print(spec_data.analysis_results("f01"))


The instance of ``calibrations`` has been automatically updated with the measured
frequency, as shown below. In addition to the columns shown below, ``calibrations`` also
stores the group to which a value belongs, whether a value is valid or not, and the
experiment id that produced a value.

.. jupyter-execute::

    pd.DataFrame(**cals.parameters_table(qubit_list=[qubit]))[columns_to_show]
    
.. _Rabi Calibration:

Calibrating the pulse amplitudes with a Rabi experiment
-------------------------------------------------------

In the Rabi experiment we apply a pulse at the frequency of the qubit 
and scan its amplitude to find the amplitude that creates a rotation 
of a desired angle. We do this with the calibration experiment :class:`.RoughXSXAmplitudeCal`.
This is a specialization of the :class:`.Rabi` experiment that will update the calibrations 
for both the :math:`X` pulse and the :math:`SX` pulse using a single experiment.

.. jupyter-execute:: 

    from qiskit_experiments.library.calibration import RoughXSXAmplitudeCal
    rabi = RoughXSXAmplitudeCal((qubit,), cals, backend=backend, amplitudes=np.linspace(-0.1, 0.1, 51))

The rough amplitude calibration is therefore a Rabi experiment in which 
each circuit contains a pulse with a gate. Different circuits correspond to pulses 
with different amplitudes.

.. jupyter-execute::

    rabi.circuits()[0].draw(output="mpl", style="iqp")

After the experiment completes the value of the amplitudes in the calibrations 
will automatically be updated. This behaviour can be controlled using the ``auto_update``
argument given to the calibration experiment at initialization.

.. jupyter-execute::

    rabi_data = rabi.run().block_for_results()
    rabi_data.figure(0)

.. jupyter-execute::

    print(rabi_data.analysis_results("rabi_rate"))

.. jupyter-execute::

    pd.DataFrame(**cals.parameters_table(qubit_list=[qubit, ()], parameters="amp"))[columns_to_show]

The table above shows that we have now updated the amplitude of our :math:`\pi` pulse 
from 0.5 to the value obtained in the most recent Rabi experiment. 
Importantly, since we linked the amplitudes of the ``x`` and ``y`` schedules 
we will see that the amplitude of the ``y`` schedule has also been updated 
as seen when requesting schedules from the :class:`.Calibrations` instance. 
Furthermore, we used the result from the Rabi experiment to also update 
the value of the ``sx`` pulse. 

.. jupyter-execute::

    cals.get_schedule("sx", qubit)

.. jupyter-execute::

    cals.get_schedule("x", qubit)
   
.. jupyter-execute::

    cals.get_schedule("y", qubit)

Saving and loading calibrations
-------------------------------

The values of the calibrated parameters can be saved to a .csv file 
and reloaded at a later point in time. 

.. jupyter-input::

    cals.save(file_type="csv", overwrite=True, file_prefix="PulseBackend")

After saving the values of the parameters you may restart your kernel. If you do so, 
you will only need to run the following cell to recover the state of your calibrations. 
Since the schedules are currently not stored we need to call our ``setup_cals`` function 
or use a library to populate an instance of Calibrations with the template schedules. 
By contrast, the value of the parameters will be recovered from the file.

.. jupyter-input::

    cals = Calibrations.from_backend(backend, library)
    cals.load_parameter_values(file_name="PulseBackendparameter_values.csv")

.. jupyter-execute::

    pd.DataFrame(**cals.parameters_table(qubit_list=[qubit, ()], parameters="amp"))[columns_to_show]

.. _DRAG Calibration:

Calibrating the value of the DRAG coefficient
---------------------------------------------

A Derivative Removal by Adiabatic Gate (DRAG) pulse is designed to minimize leakage 
and phase errors to a neighbouring transition. It is a standard pulse with an additional 
derivative component. It is designed to reduce the frequency spectrum of a 
normal pulse near the  :math:`|1\rangle - |2\rangle` transition, 
reducing the chance of leakage to the :math:`|2\rangle` state. 
The optimal value of the DRAG parameter is chosen to minimize both 
leakage and phase errors resulting from the AC Stark shift. 
The pulse envelope is :math:`f(t)=\Omega_x(t)+j\beta\frac{\rm d}{{\rm d}t}\Omega_x(t)`.
Here, :math:`\Omega_x(t)` is the envelop of the in-phase component 
of the pulse and :math:`\beta` is the strength of the quadrature 
which we refer to as the DRAG parameter and seek to calibrate 
in this experiment. The DRAG calibration will run several 
series of circuits. In a given circuit a Rp(β) - Rm(β) block
is repeated :math:`N` times. Here, Rp is a rotation 
with a positive angle and Rm is the same rotation with a 
negative amplitude.

.. jupyter-execute::

    from qiskit_experiments.library import RoughDragCal
    cal_drag = RoughDragCal([qubit], cals, backend=backend, betas=np.linspace(-20, 20, 25))
    cal_drag.set_experiment_options(reps=[3, 5, 7])
    cal_drag.circuits()[5].draw(output="mpl", style="iqp")

.. jupyter-execute::

    drag_data = cal_drag.run().block_for_results()
    drag_data.figure(0) 

.. jupyter-execute::

    print(drag_data.analysis_results("beta"))

.. jupyter-execute::

    pd.DataFrame(**cals.parameters_table(qubit_list=[qubit, ()], parameters="β"))[columns_to_show]

.. _fine-amplitude-cal:

Fine calibrations of a pulse amplitude
--------------------------------------

The amplitude of a pulse can be precisely calibrated using error amplifying gate
sequences. These gate sequences apply the same gate a variable number of times.
Therefore, if each gate has a small error :math:`d\theta` in the rotation angle then a
sequence of :math:`n` gates will have a rotation error of :math:`n` * :math:`d\theta`.
The :class:`.FineAmplitude` experiment and its subclass experiments implements these
sequences to obtain the correction value of imperfect pulses. We will first examine how
to detect imperfect pulses using the characterization version of these experiments, then
update calibrations with a calibration experiment.

.. jupyter-execute:: 

    from qiskit.pulse import InstructionScheduleMap
    from qiskit_experiments.library import FineXAmplitude

Detecting over- and under-rotated pulses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We now run the error amplifying experiments with our own pulse schedules on which we
purposefully add over- and under-rotations to observe their effects. To do this, we
create an instruction to schedule map which we populate with the schedules we wish to
work with. This instruction schedule map is then given to the transpile options of the
experiment so that the Qiskit transpiler can attach the pulse schedules to the gates in
the experiments. We base all our pulses on the default :math:`X` pulse of
:class:`.SingleTransmonTestBackend`.

.. jupyter-execute::

    x_pulse = backend.instruction_schedule_map.get('x', (qubit,)).instructions[0][1].pulse
    d0, inst_map = pulse.DriveChannel(qubit), pulse.InstructionScheduleMap()


We now take the ideal :math:`X` pulse amplitude reported by the backend and add/subtract
a 2% over/underrotation to it by scaling the ideal amplitude and see if the experiment
can detect this over/underrotation. We replace the default :math:`X` pulse in the
instruction schedule map with this over/under-rotated pulse.

.. jupyter-execute::

    ideal_amp = x_pulse.amp
    over_amp = ideal_amp*1.02
    under_amp = ideal_amp*0.98
    print(f"The reported amplitude of the X pulse is {ideal_amp:.4f} which we set as ideal_amp.") 
    print(f"we use {over_amp:.4f} amplitude for overrotation pulse and {under_amp:.4f} for underrotation pulse.")
    # build the over rotated pulse and add it to the instruction schedule map
    with pulse.build(backend=backend, name="x") as x_over:
        pulse.play(pulse.Drag(x_pulse.duration, over_amp, x_pulse.sigma, x_pulse.beta), d0)
    inst_map.add("x", (qubit,), x_over)

Let's look at one of the circuits of the :class:`.FineXAmplitude` experiment. To
calibrate the :math:`X` gate, we add an :math:`SX` gate before the :math:`X` gates to
move the ideal population to the equator of the Bloch sphere where the sensitivity to
over/under rotations is the highest.

.. jupyter-execute::
    
    overamp_exp = FineXAmplitude((qubit,), backend=backend)
    overamp_exp.set_transpile_options(inst_map=inst_map)
    overamp_exp.circuits()[4].draw(output="mpl", style="iqp")

.. jupyter-execute::

    # do the experiment
    exp_data_over = overamp_exp.run(backend).block_for_results()
    exp_data_over.figure(0)

The ping-pong pattern on the figure indicates an over-rotation which makes the initial
state rotate more than :math:`\pi`.

We now look at a pulse with an under rotation to see how the :class:`.FineXAmplitude`
experiment detects this error. We will compare the results to the over-rotation above.

.. jupyter-execute::

    # build the under rotated pulse and add it to the instruction schedule map
    with pulse.build(backend=backend, name="x") as x_under:
        pulse.play(pulse.Drag(x_pulse.duration, under_amp, x_pulse.sigma, x_pulse.beta), d0)
    inst_map.add("x", (qubit,), x_under)

    # do the experiment
    underamp_exp = FineXAmplitude((qubit,), backend=backend)
    underamp_exp.set_transpile_options(inst_map=inst_map)
        
    exp_data_under = underamp_exp.run(backend).block_for_results()
    exp_data_under.figure(0)

Similarly to the over-rotation, the under-rotated pulse creates qubit populations that
do not lie on the equator of the Bloch sphere. However, compared to the ping-pong
pattern of the over rotated pulse, the under rotated pulse produces an inverted
ping-pong pattern. This allows us to determine not only the magnitude of the rotation
error but also its sign.

.. jupyter-execute::
    
    # analyze the results
    target_angle = np.pi
    dtheta_over = exp_data_over.analysis_results("d_theta").value.nominal_value
    scale_over = target_angle / (target_angle + dtheta_over)
    dtheta_under = exp_data_under.analysis_results("d_theta").value.nominal_value
    scale_under = target_angle / (target_angle + dtheta_under)
    print(f"The ideal angle is {target_angle:.2f} rad. We measured a deviation of {dtheta_over:.3f} rad in over-rotated pulse case.")
    print(f"Thus, scale the {over_amp:.4f} pulse amplitude by {scale_over:.3f} to obtain {over_amp*scale_over:.5f}.")
    print(f"On the other hand, we measured a deviation of {dtheta_under:.3f} rad in under-rotated pulse case.")
    print(f"Thus, scale the {under_amp:.4f} pulse amplitude by {scale_under:.3f} to obtain {under_amp*scale_under:.5f}.")


Calibrating a :math:`\pi`/2 :math:`X` pulse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now we apply the same principles to a different example using the calibration version of
a Fine Amplitude experiment. The amplitude of the :math:`SX` gate, which is an :math:`X`
pulse with half the amplitude, is calibrated with the :class:`.FineSXAmplitudeCal`
experiment. Unlike the :class:`.FineSXAmplitude` experiment, the
:class:`.FineSXAmplitudeCal` experiment does not require other gates than the :math:`SX`
gate since the number of repetitions can be chosen such that the ideal population is
always on the equator of the Bloch sphere. To demonstrate the
:class:`.FineSXAmplitudeCal` experiment, we create a :math:`SX` pulse by dividing the
amplitude of the X pulse by two. We expect that this pulse might have a small rotation
error which we want to correct.

.. jupyter-execute::

    from qiskit_experiments.library import FineSXAmplitudeCal

    amp_cal = FineSXAmplitudeCal((qubit,), cals, backend=backend, schedule_name="sx")
    amp_cal.circuits()[4].draw(output="mpl", style="iqp")

Let's run the calibration experiment:

.. jupyter-execute::

    exp_data_x90p = amp_cal.run().block_for_results()
    exp_data_x90p.figure(0)

Observe, once again, that the calibrations have automatically been updated.

.. jupyter-execute::

    pd.DataFrame(**cals.parameters_table(qubit_list=[qubit, ()], parameters="amp"))[columns_to_show]

.. jupyter-execute::

    cals.get_schedule("sx", qubit)

If we run the experiment again, we expect to see that the updated calibrated gate will
have a smaller :math:`d\theta` error:

.. jupyter-execute::

    exp_data_x90p_rerun = amp_cal.run().block_for_results()
    exp_data_x90p_rerun.figure(0)

See also
--------

* API documentation: :mod:`~qiskit_experiments.calibration_management` and :mod:`~qiskit_experiments.library.calibration`
* Qiskit Textbook: `Calibrating Qubits with Qiskit Pulse <https://github.com/Qiskit/textbook/blob/main/notebooks/quantum-hardware-pulses/calibrating-qubits-pulse.ipynb>`__



