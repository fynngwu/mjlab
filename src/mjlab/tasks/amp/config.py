from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AmpStyleCfg:
  motion_path: str
  reward_weight: float = 1.0
  reward_scale: float = 5.0
  loss_scale: float = 5.0
  history: int = 2
  batch_size: int = 4096
  replay_size: int = 200_000
  learning_rate: float = 5.0e-5
  weight_decay: float = 1.0e-4
  logit_reg: float = 0.05
  grad_penalty: float = 5.0
  disc_obs_buffer_size: int = 48
