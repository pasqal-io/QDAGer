import logging
from dataclasses import replace
from typing import Any, Literal

import networkx as nx
import numpy as np
import pulser
import torch
from emu_mps import CorrelationMatrix, MPSBackend, MPSConfig, StateResult
from emu_sv import SVBackend, SVConfig
from emu_sv.utils import index_to_bitstring
from pulser import Pulse
from pulser.backend import Results
from pulser.devices import AnalogDevice
from pulser.sequence import Sequence

from sym_graphs.utils.pulse_utils import build_MIS_pulse
from sym_graphs.utils.utils import DotDict, bitstrings_to_array, factors


class Correlator:
    """Computes a k-body correlation tensor from an emu-SV result object"""

    def __init__(self, G: nx.Graph | None = None, cfg: DotDict | None = None) -> None:
        # enforce: either both None or both provided
        if (G is None) ^ (cfg is None):
            raise ValueError("Provide both G and cfg together, or neither.")

        # defaults when no args
        if G is None:
            G = nx.Graph()
            cfg = DotDict()  # adjust to the default config factory

        self.cfg = cfg
        self.adj = nx.to_numpy_array(G, nodelist=list(range(max(G.nodes(), default=-1) + 1)))
        self.type = getattr(self.cfg.pulse, "pulse_type", None)

        reg = pulser.register.Register.rectangle(
            *factors(self.adj.shape[0]),
            spacing=5,
        )
        device = AnalogDevice
        dur = getattr(self.cfg.pulse, "pulse_duration", None)
        if dur is not None and dur > device.max_sequence_duration:
            device = replace(device.to_virtual(), max_sequence_duration=dur)

        device = replace(device, max_radial_distance=int(1e6))

        seq = Sequence(reg, device)
        seq.declare_channel("ryd_glob", "rydberg_global")
        self.seq = seq

    def emulate_non_UD(
        self,
        emulator: Literal["SV", "MPS"],
        return_state: bool = True,
        verbose: bool = True,
    ) -> Results:
        """Emulate an Ising system without a specific register embedding
        Args:
            emulator (str): The type of emulator to consider
            return_state (bool) : Only relevant with SV. If True returns the
            state vector, else returns the correlation matrix
            verbose (bool) : If False, hides information about timesteps
        Raises:
            ValueError: If the pulse type initiated in the config file is not recognized
        Returns:
            Results: the state dict as a function of time
        """
        logging_level = None if verbose else logging.WARN
        Omega = self.cfg.pulse.Omega_max

        if self.type == "MIS":
            pulse = build_MIS_pulse(
                self.cfg.pulse.pulse_duration,
                self.cfg.pulse.pulse_duration / 3,
                self.cfg.pulse.pulse_duration / 3,
                Omega,
                -self.cfg.pulse.delta,
                self.cfg.pulse.delta,
            )
        elif self.type == "const":
            pulse = Pulse.ConstantPulse(
                duration=self.cfg.pulse.pulse_duration,
                amplitude=Omega,
                detuning=self.cfg.pulse.delta,
                phase=0,
            )
            print(pulse)
        else:
            raise ValueError(" Pulse type not implemented")
        if pulse is not None:
            self.seq.add(pulse, "ryd_glob", protocol="no-delay")

        N_steps = int(self.cfg.pulse.pulse_duration / self.cfg.pulse.DT)

        if emulator == "SV":
            state = StateResult(evaluation_times=[i / N_steps for i in range(N_steps + 1)])
            correlation_matrix = CorrelationMatrix(
                evaluation_times=[i / N_steps for i in range(N_steps + 1)],
            )

            if return_state:
                Obs_list = [state]
            else:
                Obs_list = [correlation_matrix]

            svconfig = SVConfig(
                dt=self.cfg.pulse.DT,
                observables=Obs_list,
                interaction_matrix=self.adj.tolist(),
                gpu=True,
                max_krylov_dim=self.cfg.SV_config.max_krylov_dim,
                krylov_tolerance=self.cfg.SV_config.krylov_tolerance,
                log_level=logging_level,
            )
            sim = SVBackend(self.seq, config=svconfig)
            return sim.run()

        if emulator == "MPS":
            correlation_matrix = CorrelationMatrix(
                evaluation_times=[i / N_steps for i in range(N_steps + 1)],
            )
            # create MPS config with custom interaction matrix,
            # ignores the register if the adjacency is specified
            mpsconfig = MPSConfig(
                dt=self.cfg.pulse.DT,
                observables=[correlation_matrix],
                interaction_matrix=self.adj.tolist(),
                max_bond_dim=self.cfg.MPS_config.max_bond_dim,
                extra_krylov_tolerance=self.cfg.MPS_config.extra_krylov_tolerance,
                num_gpus_to_use=self.cfg.MPS_config.num_gpus_to_use,
                precision=self.cfg.MPS_config.precision,
                log_level=logging_level,
            )
            sim = MPSBackend(sequence=self.seq, config=mpsconfig)
            return sim.run()

        raise ValueError("Non recognized emulator")

    def k_body_full_chunked(
        self,
        res: Any,
        k: int,
        chunk_s: int = 2048,
        chunk_t: int = 64,
        *,
        complex_t: torch.dtype = torch.complex128,  # upgraded default
        real_t: torch.dtype = torch.float64,  # upgraded default
        use_legacy_A: bool = True,
        legacy_is_msb_first: bool = False,  # set True if bitstrings were MSB-first
    ) -> torch.Tensor:
        """
        Full k-body correlator with bounded memory, chunked over S and T.
        Explicitly uses legacy `bitstrings_to_array` when `use_legacy_A=True`.

        Corr_k[t, i1, ..., ik] = Σ_s |PSI[t, s]|^2 * Π_{d=1..k} A[s, i_d]

        Parameters
        ----------
        res : Any
            Object with `.state[t].vector` (length S). Vectors are treated as **complex128** by default.
        k : int
            Body order (>=1).
        chunk_s : int, default 2048
            Basis states per chunk.
        chunk_t : int, default 64
            Time steps per chunk.
        complex_t : torch.dtype, keyword-only, default torch.complex128
            Complex dtype used for input vectors.
        real_t : torch.dtype, keyword-only, default torch.float64
            Real dtype used for outputs and for casting A / probabilities.
        use_legacy_A : bool, default True
            Build A via `bitstrings_to_array`.
        legacy_is_msb_first : bool, default False
            Set True if legacy bit order was MSB-first.

        Returns
        -------
        torch.Tensor
            Tensor of shape (T, N, ..., N) (k axes), dtype=`real_t` (default float64),
            on the same device as the input vectors.

        Notes
        -----
        - This version upgrades the numeric path to complex128 → float64 by default.
        - If inputs are not tensors, they are converted and cast to `complex_t`.
        - Probabilities |ψ|^2 are accumulated in `real_t`.
        """
        assert k >= 1
        N: int = self.adj.shape[0]

        # Device & dimensions
        v0 = res.state[0].vector
        device = v0.device if isinstance(v0, torch.Tensor) else torch.device("cpu")
        S: int = v0.numel() if isinstance(v0, torch.Tensor) else int(np.asarray(v0).size)
        T: int = len(res.state)

        # ---- Build A (S, N) explicitly using a legacy function if requested ----
        if use_legacy_A:
            # can be heavy for larger N.
            bits = [index_to_bitstring(N, i) for i in range(S)]
            A_cpu = bitstrings_to_array(bits)  # expected shape (S, N) with {0,1}
            A = torch.as_tensor(A_cpu, dtype=real_t)  # CPU tensor first (float64 by default)
            if legacy_is_msb_first:
                # Flip to LSB-first internal convention if legacy was MSB-first
                A = torch.flip(A, dims=[1])
            A = A.to(device=device, dtype=real_t, non_blocking=True)
            A_bool = A.bool()
        else:
            # GPU-native construction (LSB-first)
            idx = torch.arange(S, device=device, dtype=torch.long).unsqueeze(1)  # (S,1)
            shifts = torch.arange(N, device=device, dtype=torch.long).unsqueeze(0)  # (1,N)
            A_bool = ((idx >> shifts) & 1) > 0  # (S,N) bool
            # If MSB-first instead:
            # A_bool = torch.flip(A_bool, dims=[1])

        # Output accumulator
        out = torch.zeros((T,) + (N,) * k, device=device, dtype=real_t)
        out_flat = out.view(T, -1)  # (T, N**k)

        # ---- Chunk over S ----
        for s0 in range(0, S, chunk_s):
            s1 = min(s0 + chunk_s, S)
            B = A_bool[s0:s1, :]  # (S_c, N) bool

            # Build M via k-fold outer in bool, then cast once to real_t
            M = B
            for r in range(1, k):
                M = M.unsqueeze(-1) & B.view(B.size(0), *([1] * r), N)
            M = (
                M.reshape(B.size(0), -1).to(device=device, dtype=real_t).contiguous()
            )  # (S_c, N**k)

            # ---- Stream over T ----
            for t0 in range(0, T, chunk_t):
                t1 = min(t0 + chunk_t, T)

                # Assemble P (T_c, S_c)
                Ps = []
                for t in range(t0, t1):
                    vec = res.state[t].vector
                    if not isinstance(vec, torch.Tensor):
                        vec = torch.from_numpy(np.asarray(vec))
                    # Enforce upgraded complex dtype on the fly
                    v_chunk = vec.to(device=device, dtype=complex_t, non_blocking=True)[s0:s1]
                    Ps.append((v_chunk * v_chunk.conj()).real.to(real_t))
                P = torch.stack(Ps, dim=0).contiguous()  # (T_c, S_c)

                # Accumulate: (T_c, S_c) @ (S_c, N**k) -> (T_c, N**k)
                out_flat[t0:t1].addmm_(P, M)

        return out
