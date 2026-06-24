from __future__ import annotations

import numpy as np
import torch

from .math_utils import covariance_to_features
from .smc import Posterior


def build_metadata(
    n_qubits: int,
    m_evo: int,
    theta_dim: int,
    cov_feat_dim: int,
    use_smc_feedback: bool,
    belief_mode: str,
    device: torch.device,
    info: dict[str, object] | None = None,
    posterior: Posterior | None = None,
    shots_max: int = 1,
) -> torch.Tensor:
    action_meta_dim = n_qubits + 3 + m_evo + 1
    if not use_smc_feedback:
        belief_mode = "none"
    if belief_mode not in {"both", "mean", "cov", "none"}:
        raise ValueError(f"Unknown belief_mode: {belief_mode}")
    belief_dim = 0
    if belief_mode in {"both", "mean"}:
        belief_dim += theta_dim
    if belief_mode in {"both", "cov"}:
        belief_dim += cov_feat_dim
    metadata = torch.zeros(action_meta_dim + belief_dim, device=device)

    if info is not None:
        qi = int(info["qubit_idx"])
        bi = int(info["basis_idx"])
        ti = int(info["time_idx"])
        shots = int(info.get("shots", shots_max))
        metadata[qi] = 1.0
        metadata[n_qubits + bi] = 1.0
        metadata[n_qubits + 3 + ti] = 1.0
        metadata[n_qubits + 3 + m_evo] = float(np.log2(max(1, shots)) / np.log2(max(2, shots_max)))

    if belief_mode != "none" and posterior is not None:
        start = action_meta_dim
        if belief_mode in {"both", "mean"}:
            metadata[start : start + theta_dim] = posterior.mean.detach()
            start += theta_dim
        if belief_mode in {"both", "cov"}:
            metadata[start : start + cov_feat_dim] = covariance_to_features(posterior.cov, max_eigs=8).detach()

    return metadata
