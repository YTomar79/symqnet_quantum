from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden: int = 128):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.enc_fc1 = nn.Linear(self.input_dim, hidden)
        self.enc_fc2 = nn.Linear(hidden, hidden)
        self.enc_mu = nn.Linear(hidden, self.latent_dim)
        self.enc_logvar = nn.Linear(hidden, self.latent_dim)
        self.dec_fc1 = nn.Linear(self.latent_dim, hidden)
        self.dec_fc2 = nn.Linear(hidden, hidden)
        self.dec_out = nn.Linear(hidden, self.input_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.enc_fc1(x))
        h = F.relu(self.enc_fc2(h))
        return self.enc_mu(h), self.enc_logvar(h).clamp(-10.0, 10.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        return self.dec_out(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


def vae_loss(recon: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float) -> torch.Tensor:
    recon_loss = F.mse_loss(recon, target, reduction="sum") / target.shape[0]
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.shape[0]
    return recon_loss + beta * kl
