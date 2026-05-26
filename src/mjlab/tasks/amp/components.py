from __future__ import annotations

import torch
from torch import nn


class AMPDiscriminator(nn.Module):
  def __init__(self, obs_dim: int):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(obs_dim, 1024),
      nn.LeakyReLU(0.2),
      nn.Linear(1024, 512),
      nn.LeakyReLU(0.2),
      nn.Linear(512, 1),
    )

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.net(obs).squeeze(-1)


class AMPReplayBuffer:
  def __init__(self, capacity: int, obs_dim: int, device: str | torch.device):
    self.obs = torch.empty((capacity, obs_dim), device=device)
    self.commands = torch.empty((capacity, 3), device=device)
    self.capacity = capacity
    self.ptr = 0
    self.size = 0
    self.device = device

  def add(self, obs: torch.Tensor, commands: torch.Tensor) -> None:
    obs = obs.detach()
    commands = commands[:, :3].detach()
    count = obs.shape[0]
    if count >= self.capacity:
      self.obs[:] = obs[-self.capacity :]
      self.commands[:] = commands[-self.capacity :]
      self.ptr = 0
      self.size = self.capacity
      return
    end = self.ptr + count
    if end <= self.capacity:
      self.obs[self.ptr : end] = obs
      self.commands[self.ptr : end] = commands
    else:
      first = self.capacity - self.ptr
      self.obs[self.ptr :] = obs[:first]
      self.commands[self.ptr :] = commands[:first]
      self.obs[: end % self.capacity] = obs[first:]
      self.commands[: end % self.capacity] = commands[first:]
    self.ptr = end % self.capacity
    self.size = min(self.size + count, self.capacity)

  def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    indexes = torch.randint(self.size, (batch_size,), device=self.device)
    return self.obs[indexes], self.commands[indexes]
