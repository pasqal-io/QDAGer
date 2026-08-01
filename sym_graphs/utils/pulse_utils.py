import math

import numpy as np
import pulser
import torch
from pulser.waveforms import CompositeWaveform, RampWaveform

from sym_graphs.utils.utils import _round_nearest


def build_MIS_pulse(
    duration: float,
    rise_duration: float,
    fall_duration: float,
    amplitude_maximum: float | None = 1,
    detuning_minimum: float | None = -1,
    detuning_maximum: float | None = 1,
    clock_period: int | None = 4,
    **params_dict: dict,
) -> pulser.Pulse:
    """
    Builds a standard annealing pulse for MIS with variable duration segments

    Args:
        duration: duration of the full pulse in ns
        rise_duration: duration of initial amplitude ramp up in ns
        fall_duration: duration of final amplitude ramp down in ns
        amplitude_maximum: maximum reached by amplitude (default 1)
        detuning_minimum: starting point of detuning (default -1)
        detuning_maximum: finishing point of detuning (default 1)

    Returns:
        MIS pulse pl.Pulse
    """
    assert clock_period is not None
    duration = _round_nearest(duration, clock_period)
    rise_duration = _round_nearest(rise_duration, clock_period)
    fall_duration = _round_nearest(fall_duration, clock_period)
    sweep_duration = duration - rise_duration - fall_duration

    rise = pulser.Pulse.ConstantDetuning(
        RampWaveform(rise_duration, 0.0, amplitude_maximum),
        detuning_minimum,
        0.0,
    )
    sweep = pulser.Pulse.ConstantAmplitude(
        amplitude_maximum,
        RampWaveform(sweep_duration, detuning_minimum, detuning_maximum),
        0.0,
    )
    fall = pulser.Pulse.ConstantDetuning(
        RampWaveform(fall_duration, amplitude_maximum, 0.0),
        detuning_maximum,
        0.0,
    )
    amp = CompositeWaveform(rise.amplitude, sweep.amplitude, fall.amplitude)
    det = CompositeWaveform(rise.detuning, sweep.detuning, fall.detuning)

    return pulser.Pulse(amp, det, 0)


def pulser_afm_sequence_from_register(
    reg: pulser.Register,
    Omega_max: float,
    delta_0: float,
    delta_f: float,
    t_rise: float,
    t_fall: float,
    device: pulser.devices = pulser.devices.MockDevice,
):
    t_sweep = (delta_f - delta_0) / (2 * np.pi * 10) * 1000

    rise = pulser.Pulse.ConstantDetuning(
        pulser.waveforms.RampWaveform(t_rise, 0.0, Omega_max),
        delta_0,
        0.0,
    )
    sweep = pulser.Pulse.ConstantAmplitude(
        Omega_max,
        pulser.waveforms.RampWaveform(t_sweep, delta_0, delta_f),
        0.0,
    )
    fall = pulser.Pulse.ConstantDetuning(
        pulser.waveforms.RampWaveform(t_fall, Omega_max, 0.0),
        delta_f,
        0.0,
    )

    seq = pulser.Sequence(reg, device)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(rise, "ising_global")
    seq.add(sweep, "ising_global")
    seq.add(fall, "ising_global")

    return seq


def pulser_afm_sequence_ring(
    num_qubits: int,
    Omega_max: float,
    U: float,
    delta_0: float,
    delta_f: float,
    t_rise: float,
    t_fall: float,
    device: pulser.devices = pulser.devices.MockDevice,
):
    # Define a ring of atoms distanced by a blockade radius distance:
    R_interatomic = device.rydberg_blockade_radius(U)
    coords = (
        R_interatomic
        / (2 * math.tan(math.pi / num_qubits))
        * torch.tensor(
            [
                [
                    math.cos(theta * 2 * math.pi / num_qubits),
                    math.sin(theta * 2 * math.pi / num_qubits),
                ]
                for theta in range(num_qubits)
            ],
        )
    )

    reg = pulser.Register.from_coordinates(coords, prefix="q")

    return pulser_afm_sequence_from_register(
        reg,
        Omega_max=Omega_max,
        delta_0=delta_0,
        delta_f=delta_f,
        t_rise=t_rise,
        t_fall=t_fall,
    )
