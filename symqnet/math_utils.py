from __future__ import annotations

import math

import torch


@torch.no_grad()
def mean_outcome_from_state(
    psi: torch.Tensor,
    flip_idx: torch.Tensor,
    sign01: torch.Tensor,
    phase: torch.Tensor,
    basis_idx: int,
) -> torch.Tensor:
    """Mean +/-1 outcome for X/Y/Z using the codebase convention |1> -> +1."""
    psi_flip = psi[..., flip_idx]
    if basis_idx == 2:
        probs = (psi.abs() ** 2).to(sign01.dtype)
        return (probs * sign01).sum(dim=-1)
    if basis_idx == 0:
        ex = (psi.conj() * psi_flip).sum(dim=-1).real
        return -ex
    if basis_idx == 1:
        i_unit = torch.tensor(1j, device=psi.device, dtype=psi.dtype)
        ey = (psi.conj() * (i_unit * phase) * psi_flip).sum(dim=-1).real
        return -ey
    raise ValueError("basis_idx must be 0 (X), 1 (Y), or 2 (Z)")


def covariance_to_features(cov: torch.Tensor, max_eigs: int = 8) -> torch.Tensor:
    diag = torch.log(torch.diag(cov) + 1e-8)
    eigvals = torch.linalg.eigvalsh(cov)
    if eigvals.numel() < max_eigs:
        pad = torch.zeros(max_eigs - eigvals.numel(), device=cov.device, dtype=cov.dtype)
        topk = torch.cat([pad, eigvals])
    else:
        topk = eigvals[-max_eigs:]
    return torch.cat([diag, topk], dim=0)


def gaussian_entropy_from_cov(cov: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    d = int(cov.shape[0])
    eye = torch.eye(d, device=cov.device, dtype=cov.dtype)
    sign, logdet = torch.linalg.slogdet(cov + eps * eye)
    if (sign <= 0).any():
        sign, logdet = torch.linalg.slogdet(cov + 1e-6 * eye)
    return 0.5 * (d * (1.0 + math.log(2.0 * math.pi)) + logdet)


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
