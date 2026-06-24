from __future__ import annotations

import torch
import torch.nn as nn


class TemporalContextualAggregator(nn.Module):
    def __init__(self, width: int, history: int, num_heads: int = 2, dropout: float = 0.1, layers: int = 2):
        super().__init__()
        self.width = int(width)
        self.history = int(history)
        self.attn_dim = ((self.width + num_heads - 1) // num_heads) * num_heads
        self.pre = nn.Linear(self.width, self.attn_dim, bias=False) if self.attn_dim != self.width else nn.Identity()
        self.post = nn.Linear(self.attn_dim, self.width, bias=False) if self.attn_dim != self.width else nn.Identity()
        self.pos_emb = nn.Parameter(torch.zeros(1, self.history, self.attn_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.attn_dim,
            nhead=num_heads,
            dim_feedforward=4 * self.attn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.out = nn.Sequential(nn.Linear(self.width, self.width), nn.LayerNorm(self.width))

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        single = window.dim() == 2
        if single:
            window = window.unsqueeze(0)
        B, T, D = window.shape
        if D != self.width:
            raise ValueError(f"Expected feature width {self.width}, got {D}")
        if T < self.history:
            pad = torch.zeros(B, self.history - T, D, device=window.device, dtype=window.dtype)
            window = torch.cat([pad, window], dim=1)
        elif T > self.history:
            window = window[:, -self.history :, :]
        h = self.pre(window) + self.pos_emb[:, : window.shape[1], :]
        mask = torch.triu(torch.full((h.shape[1], h.shape[1]), float("-inf"), device=h.device), diagonal=1)
        h = self.encoder(h, mask=mask)
        out = self.out(self.post(h[:, -1, :]))
        return out.squeeze(0) if single else out


class LastStepAggregator(nn.Module):
    def __init__(self, width: int, history: int):
        super().__init__()
        self.width = int(width)
        self.history = int(history)
        self.out = nn.Sequential(nn.Linear(self.width, self.width), nn.ReLU(), nn.LayerNorm(self.width))

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        single = window.dim() == 2
        if single:
            window = window.unsqueeze(0)
        out = self.out(window[:, -1, :])
        return out.squeeze(0) if single else out
