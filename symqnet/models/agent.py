from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .graph import GraphEmbed, NoGraphEmbed, make_chain_edges, make_random_edges, make_star_edges
from .temporal import LastStepAggregator, TemporalContextualAggregator


class PolicyValueHead(nn.Module):
    def __init__(self, width: int, n_actions: int):
        super().__init__()
        hidden = 2 * int(width)
        self.trunk = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        h = self.trunk(x)
        dist = Categorical(logits=self.policy(h))
        value = self.value(h).squeeze(-1)
        return dist, value.squeeze(0) if single else value


class SymQNetAgent(nn.Module):
    """Action policy/value network. Parameter estimates come from SMC feedback, not a learned estimator head."""

    def __init__(
        self,
        vae: nn.Module,
        n_qubits: int,
        latent_dim: int,
        history: int,
        n_actions: int,
        m_evo: int,
        gnn_layers: int = 2,
        graph: str = "chain",
        temporal: str = "transformer",
        use_smc_feedback: bool = True,
        belief_mode: str = "both",
        use_vae: bool = True,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.vae = vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        self.n_qubits = int(n_qubits)
        self.use_vae = bool(use_vae)
        self.latent_dim = int(latent_dim) if self.use_vae else self.n_qubits
        self.raw_obs_dim = self.n_qubits
        self.history = int(history)
        self.theta_dim = 2 * self.n_qubits - 1
        self.cov_feat_dim = self.theta_dim + 8
        self.action_meta_dim = self.n_qubits + 3 + int(m_evo) + 1
        self.use_smc_feedback = bool(use_smc_feedback)
        self.belief_mode = "none" if not self.use_smc_feedback else str(belief_mode)
        if self.belief_mode not in {"both", "mean", "cov", "none"}:
            raise ValueError(f"Unknown belief_mode: {self.belief_mode}")
        self.belief_dim = 0
        if self.belief_mode in {"both", "mean"}:
            self.belief_dim += self.theta_dim
        if self.belief_mode in {"both", "cov"}:
            self.belief_dim += self.cov_feat_dim
        self.meta_dim = self.action_meta_dim + self.belief_dim
        self.width = self.raw_obs_dim + self.latent_dim + self.meta_dim
        self.node_local_dim = self.n_qubits + 8

        if graph == "none":
            self.graph_embed = NoGraphEmbed(self.width)
        else:
            if graph == "star":
                edge_index, edge_attr = make_star_edges(self.n_qubits, device)
            elif graph == "random":
                edge_index, edge_attr = make_random_edges(self.n_qubits, device)
            else:
                edge_index, edge_attr = make_chain_edges(self.n_qubits, device)
            self.graph_embed = GraphEmbed(self.n_qubits, self.width + self.node_local_dim, self.width, edge_index, edge_attr, gnn_layers)

        if temporal == "last":
            self.temporal = LastStepAggregator(self.width, self.history)
        else:
            self.temporal = TemporalContextualAggregator(self.width, self.history)
        self.policy_value = PolicyValueHead(self.width, n_actions)
        self._history: list[torch.Tensor] = []

    def reset_buffer(self) -> None:
        self._history.clear()

    def encode_observation(self, obs: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        z = self.encode_observations(obs)
        return torch.cat([obs, z, metadata], dim=-1)

    def encode_observations(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.use_vae:
            return obs
        with torch.no_grad():
            z, _ = self.vae.encode(obs.unsqueeze(0) if obs.dim() == 1 else obs)
        return z.squeeze(0) if obs.dim() == 1 else z

    def forward(self, obs: torch.Tensor, metadata: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        z_meta = self.encode_observation(obs, metadata)
        z_g = self.graph_embed(self._node_features(z_meta)) if isinstance(self.graph_embed, GraphEmbed) else self.graph_embed(z_meta)
        self._history.append(z_g)
        if len(self._history) > self.history:
            self._history.pop(0)
        window = torch.stack(self._history, dim=0)
        c_t = self.temporal(window)
        return self.policy_value(c_t)

    def forward_window(self, z_meta_window: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        if z_meta_window.dim() == 2:
            z_meta_window = z_meta_window.unsqueeze(0)
            single = True
        else:
            single = False
        B, T, D = z_meta_window.shape
        flat = z_meta_window.reshape(B * T, D)
        if isinstance(self.graph_embed, GraphEmbed):
            z_g = self.graph_embed(self._node_features(flat)).reshape(B, T, D)
        else:
            z_g = self.graph_embed(flat).reshape(B, T, D)
        c_t = self.temporal(z_g)
        dist, value = self.policy_value(c_t)
        return dist, value.squeeze(0) if single else value

    def _node_features(self, x: torch.Tensor) -> torch.Tensor:
        """Expand a flat transition vector into node-specific qubit features."""
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        B = x.shape[0]
        device = x.device
        raw_obs = x[:, : self.raw_obs_dim]
        meta_start = self.raw_obs_dim + self.latent_dim
        metadata = x[:, meta_start:]

        action_q = metadata[:, : self.n_qubits]
        belief_start = self.action_meta_dim
        mean = torch.zeros(B, self.theta_dim, device=device, dtype=x.dtype)
        cov_diag = torch.zeros(B, self.theta_dim, device=device, dtype=x.dtype)
        if self.belief_mode in {"both", "mean"}:
            mean = metadata[:, belief_start : belief_start + self.theta_dim]
            belief_start += self.theta_dim
        if self.belief_mode in {"both", "cov"}:
            cov_diag = metadata[:, belief_start : belief_start + self.theta_dim]

        node_id = torch.eye(self.n_qubits, device=device, dtype=x.dtype).unsqueeze(0).expand(B, -1, -1)
        base = x[:, None, :].expand(B, self.n_qubits, self.width)
        obs_i = raw_obs.unsqueeze(-1)
        action_i = action_q.unsqueeze(-1)

        local_mean = torch.zeros(B, self.n_qubits, 3, device=device, dtype=x.dtype)
        local_cov = torch.zeros_like(local_mean)
        h_offset = self.n_qubits - 1
        for i in range(self.n_qubits):
            local_mean[:, i, 0] = mean[:, h_offset + i]
            local_cov[:, i, 0] = cov_diag[:, h_offset + i]
            if i > 0:
                local_mean[:, i, 1] = mean[:, i - 1]
                local_cov[:, i, 1] = cov_diag[:, i - 1]
            if i < self.n_qubits - 1:
                local_mean[:, i, 2] = mean[:, i]
                local_cov[:, i, 2] = cov_diag[:, i]

        nodes = torch.cat([base, node_id, obs_i, action_i, local_mean, local_cov], dim=-1)
        return nodes.squeeze(0) if single else nodes
