AC Stark Effect
===============

When a qubit is driven with an off-resonant tone,
the qubit frequency :math:`f_0` is slightly shifted through what is known as the (AC) Stark effect.
This technique is sometimes used to characterize qubit properties in the vicinity of
the base frequency, especially with a fixed frequency qubit architecture which otherwise
doesn't have a knob to control frequency [1]_.

The important control parameters of the Stark effect are the amplitude
:math:`\Omega` and frequency :math:`f_S` of
the off-resonant tone, which we will call the *Stark tone* in the following.
In the low power limit, the amount of frequency shift :math:`\delta f_S`
that the qubit may experience is described as follows [2]_:

.. math::

    \delta f_S \propto \frac{\alpha}{2\Delta\left(\alpha - \Delta\right)} \Omega^2,

where :math:`\alpha` is the qubit anharmonicity and :math:`\Delta=f_S - f_0` is the
frequency separation of the Stark tone from the qubit frequency :math:`f_0`.
We sometimes call :math:`\delta f_S` the *Stark shift* [3]_.


.. _stark_tone_implementation:

Stark tone implementation in Qiskit
-----------------------------------

Usually, we fix the Stark tone frequency :math:`f_S` and control the amplitude :math:`\Omega`
to modulate the qubit frequency.
In Qiskit, we often use an abstracted amplitude :math:`\bar{\Omega}`,
instead of the physical amplitude :math:`\Omega` in the experiments.

Because the Stark shift :math:`\delta f_S` has a quadratic dependence on
the tone amplitude :math:`\Omega`, the resulting shift is not sensitive to its sign.
On the other hand, the sign of the shift depends on the sign of the frequency offset :math:`\Delta`.
In a typical parameter regime of :math:`|\Delta | < | \alpha |`,

.. math::

    \text{sign}(\delta f_S) = - \text{sign}(\Delta).

In other words, positive (negative) Stark shift occurs when the tone frequency :math:`f_S`
is lower (higher) than the qubit frequency :math:`f_0`.
When an experimentalist wants to perform spectroscopy of some qubit parameter
in the vicinity of :math:`f_0`, one must manage the sign of :math:`f_S`
in addition to the magnitude of :math:`\Omega`.

To alleviate such experimental complexity, an abstracted amplitude :math:`\bar{\Omega}`
with virtual sign is introduced in Qiskit Experiments.
This works as follows:

.. math::

    \Delta &= - \text{sign}(\bar{\Omega}) | \Delta |, \\
    \Omega &= | \bar{\Omega} |.

Stark experiments in Qiskit usually take two control parameters :math:`(\bar{\Omega}, |\Delta|)`,
which are specified by ``stark_amp`` and ``stark_freq_offset`` in the experiment options, respectively.
In this representation, the sign of the Stark shift matches the sign of :math:`\bar{\Omega}`.

.. math::

    \text{sign}(\delta f_S) = \text{sign}(\bar{\Omega})

This allows an experimentalist to control both the sign and the amount of
the Stark shift with the ``stark_amp`` experiment option.
Note that ``stark_freq_offset`` should be set as a positive number.


.. _stark_frequency_consideration:

Stark tone frequency
--------------------

As you can see in the equation for :math:`\delta f_S` above,
:math:`\Delta=0` yields a singular point where :math:`\delta f_S` diverges.
This corresponds to a Rabi drive, where the qubit is driven on resonance and
coherent state exchange occurs between :math:`|0\rangle` and :math:`|1\rangle`
instead of the Stark shift.
Another frequency that should be avoided for the Stark tone is :math:`\Delta=\alpha` which
corresponds to the transition from :math:`|1\rangle` to :math:`|2\rangle`.
In the high power limit, :math:`\Delta = \alpha/2` should also be avoided since
this causes the direct excitation from :math:`|0\rangle` to :math:`|2\rangle`
through what is known as a two-photon transition.

The Stark tone frequency must be sufficiently separated from all of these frequencies
to avoid unwanted state transitions (frequency collisions).
In reality, the choice of the frequency could be even more complicated
due to the transition levels of the nearest neighbor qubits.
The frequency must be carefully chosen to avoid frequency collisions [4]_.


.. _stark_channel_consideration:

Stark tone channel
------------------

It may be necessary to supply a pulse channel to apply the Stark tone.
In Qiskit Experiments, the Stark experiments usually have an experiment option ``stark_channel``
to specify this.
By default, the Stark tone is applied to the same channel as the qubit drive
with a frequency shift. This frequency shift might update the channel frame,
which accumulates unwanted phase against the frequency difference between
the qubit drive :math:`f_0` and Stark tone frequencies :math:`f_S` in addition to
the qubit Stark shfit :math:`\delta f_s`.
You can use a dedicated Stark drive channel if available.
Otherwise, you may want to use a control channel associated with the physical
drive port of the qubit.

In a typical IBM device using the cross-resonance drive architecture,
such channel can be identified with your backend as follows:

.. jupyter-execute::

    from qiskit.providers.fake_provider import FakeHanoiV2

    backend = FakeHanoiV2()
    qubit = 0

    for qpair in backend.coupling_map:
        if qpair[0] == qubit:
            break

    print(backend.control_channel(qpair)[0])

This returns a control channel for which the qubit is the control qubit.
This approach may not work for other device architectures.


References
----------

.. [1] Malcolm Carroll, Sami Rosenblatt, Petar Jurcevic, Isaac Lauer and Abhinav Kandala,
    Dynamics of superconducting qubit relaxation times, npj Quantum Inf 8, 132 (2022).
    https://arxiv.org/abs/2105.15201

.. [2] Easwar Magesan, Jay M. Gambetta, Effective Hamiltonian models of the cross-resonance gate,
    Phys. Rev. A 101, 052308 (2020).
    https://arxiv.org/abs/1804.04073

.. [3] Wikipedia. "Autler–Townes effect" Wikipedia Foundation.
    https://en.wikipedia.org/wiki/Autler%E2%80%93Townes_effect

.. [4] Jared B. Hertzberg, Eric J. Zhang, Sami Rosenblatt, et. al.,
    Laser-annealing Josephson junctions for yielding scaled-up superconducting quantum processors,
    npj Quantum Information 7, 129 (2021).
    https://arxiv.org/abs/2009.00781