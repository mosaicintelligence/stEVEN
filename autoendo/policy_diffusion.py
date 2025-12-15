"""
policy_diffusion.py

Lightweight conditional diffusion policy for continuous actions.
Trains on (obs, action) pairs via denoising score matching and can
sample actions conditioned on flattened observations at inference.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    freqs = torch.exp(
        torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
        * -(torch.log(torch.tensor(10_000.0)) / (half_dim - 1))
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        timesteps: int = 16,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        time_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        input_dim = action_dim + obs_dim + time_embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.time_embed_dim = time_embed_dim

    def predict_noise(self, obs: torch.Tensor, action_noisy: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = _time_embedding(timesteps, self.time_embed_dim)
        model_input = torch.cat([obs, action_noisy, t_emb], dim=-1)
        return self.net(model_input)

    def loss(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        batch_size = action.shape[0]
        device = action.device
        t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        noise = torch.randn_like(action)
        alpha_bar = self.alphas_cumprod[t].unsqueeze(-1)
        noisy_action = torch.sqrt(alpha_bar) * action + torch.sqrt(1 - alpha_bar) * noise
        pred = self.predict_noise(obs, noisy_action, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, steps: Optional[int] = None, clamp: Optional[tuple[float, float]] = None) -> torch.Tensor:
        steps = steps or self.timesteps
        batch_size = obs.shape[0]
        action = torch.randn(batch_size, self.action_dim, device=obs.device)
        for i in reversed(range(steps)):
            t = torch.full((batch_size,), i, device=obs.device, dtype=torch.long)
            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_bar = self.alphas_cumprod[i]
            pred_noise = self.predict_noise(obs, action, t)
            coef1 = 1 / torch.sqrt(alpha)
            coef2 = beta / torch.sqrt(1 - alpha_bar)
            action = coef1 * (action - coef2 * pred_noise)
            if i > 0:
                noise = torch.randn_like(action)
                action = action + torch.sqrt(beta) * noise
        if clamp is not None:
            action = torch.clamp(action, clamp[0], clamp[1])
        return action
