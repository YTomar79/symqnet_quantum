from __future__ import annotations

import torch
import torch.nn as nn


def make_chain_edges(n_qubits: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    edges = [(i, i + 1) for i in range(n_qubits - 1)] + [(i + 1, i) for i in range(n_qubits - 1)]
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    edge_attr = torch.ones(len(edges), 1, dtype=torch.float32, device=device) * 0.1
    return edge_index, edge_attr


def make_star_edges(n_qubits: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    edges = [(0, i) for i in range(1, n_qubits)] + [(i, 0) for i in range(1, n_qubits)]
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    edge_attr = torch.ones(len(edges), 1, dtype=torch.float32, device=device) * 0.1
    return edge_index, edge_attr


def make_random_edges(n_qubits: int, device: torch.device | str, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    undirected = set()
    target_edges = max(1, n_qubits - 1)
    while len(undirected) < target_edges:
        src = int(torch.randint(0, n_qubits, (), generator=generator).item())
        tgt = int(torch.randint(0, n_qubits, (), generator=generator).item())
        if src != tgt:
            undirected.add(tuple(sorted((src, tgt))))
    edges = []
    for src, tgt in sorted(undirected):
        edges.extend([(src, tgt), (tgt, src)])
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    edge_attr = torch.ones(len(edges), 1, dtype=torch.float32, device=device) * 0.1
    return edge_index, edge_attr


class GraphEmbed(nn.Module):
    def __init__(
        self,
        n_qubits: int,
        input_dim: int,
        width: int,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        layers: int = 2,
    ):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.input_dim = int(input_dim)
        self.width = int(width)
        self.layers = int(layers)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_attr", edge_attr)
        self.node_in = nn.Sequential(nn.Linear(self.input_dim, self.width), nn.ReLU(), nn.LayerNorm(self.width))
        self.phi_e = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.width + 1, self.width),
                    nn.ReLU(),
                    nn.Linear(self.width, self.width),
                )
                for _ in range(self.layers)
            ]
        )
        self.phi_n = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.width, self.width),
                    nn.ReLU(),
                    nn.LayerNorm(self.width),
                    nn.Linear(self.width, self.width),
                )
                for _ in range(self.layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single = x.dim() == 2
        if single:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(f"GraphEmbed expects node features shaped (B, N, D), got {tuple(x.shape)}")
        if x.shape[1] != self.n_qubits or x.shape[2] != self.input_dim:
            raise ValueError(
                f"GraphEmbed expected node shape (N={self.n_qubits}, D={self.input_dim}), got {tuple(x.shape[1:])}"
            )
        B = x.shape[0]
        h = self.node_in(x)
        src, tgt = self.edge_index
        E = src.numel()
        edge_attr = self.edge_attr.view(1, E, 1).expand(B, E, 1)
        for phi_e, phi_n in zip(self.phi_e, self.phi_n):
            hi = h[:, src, :]
            hj = h[:, tgt, :]
            msg = phi_e(torch.cat([hi, hj, edge_attr], dim=-1))
            agg = torch.zeros_like(h)
            idx = src.view(1, E, 1).expand(B, E, self.width)
            agg.scatter_add_(dim=1, index=idx, src=msg)
            h = h + phi_n(torch.cat([h, agg], dim=-1))
        out = h.mean(dim=1)
        return out.squeeze(0) if single else out


class NoGraphEmbed(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.LayerNorm(width), nn.Linear(width, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
