from __future__ import annotations

from types import SimpleNamespace

import torch

from mjlab.tasks.amp.style_reward import AMPStyleReward


class _FixedDiscriminator:
  def __call__(self, amp_obs: torch.Tensor) -> torch.Tensor:
    return torch.tensor([1.0, -1.0, 3.0], device=amp_obs.device)


def test_amp_reward_uses_lsgan_quadratic_reward() -> None:
  reward = AMPStyleReward.__new__(AMPStyleReward)
  reward.cfg = SimpleNamespace(reward_scale=2.0)
  reward.discriminator = _FixedDiscriminator()

  amp_obs = torch.zeros((3, 202))
  actual = reward.compute_reward(amp_obs, dt=1.0)

  expected = torch.tensor([2.0, 0.0, 0.0])
  torch.testing.assert_close(actual, expected)
