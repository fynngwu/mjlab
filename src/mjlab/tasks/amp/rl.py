from __future__ import annotations

from typing import Any

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.amp.config import AmpStyleCfg
from mjlab.tasks.amp.style_reward import AMPStyleReward
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.utils.spaces import Space


class AMPRewardVecEnvWrapper(VecEnv):
  def __init__(self, env: RslRlVecEnvWrapper, cfg: AmpStyleCfg):
    self.env = env
    self.amp_style = AMPStyleReward(env.unwrapped, cfg, env.device)
    self.amp_style.reset()

    self.num_envs = env.num_envs
    self.device = env.device
    self.max_episode_length = env.max_episode_length
    self.num_actions = env.num_actions

  @property
  def cfg(self):
    return self.env.cfg

  @property
  def render_mode(self) -> str | None:
    return self.env.render_mode

  @property
  def observation_space(self) -> Space:
    return self.env.observation_space

  @property
  def action_space(self) -> Space:
    return self.env.action_space

  @property
  def unwrapped(self):
    return self.env.unwrapped

  @property
  def episode_length_buf(self) -> torch.Tensor:
    return self.env.episode_length_buf

  @episode_length_buf.setter
  def episode_length_buf(self, value: torch.Tensor) -> None:
    self.env.episode_length_buf = value

  @classmethod
  def class_name(cls) -> str:
    return cls.__name__

  def seed(self, seed: int = -1) -> int:
    return self.env.seed(seed)

  def get_observations(self) -> TensorDict:
    return self.env.get_observations()

  def reset(self) -> tuple[TensorDict, dict]:
    obs, extras = self.env.reset()
    self.amp_style.reset()
    return obs, extras

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    obs, task_reward, dones, extras = self.env.step(actions)
    amp_obs = self.amp_style.observe(dones)
    style_reward = self.amp_style.compute_reward(amp_obs)
    weighted_style_reward = self.amp_style.cfg.reward_weight * style_reward
    reward = task_reward + weighted_style_reward
    amp_metrics = self.amp_style.update(amp_obs)

    log = extras.setdefault("log", {})
    log["AMP/task_reward"] = task_reward.mean().detach()
    log["AMP/style_reward"] = style_reward.mean().detach()
    log["AMP/weighted_style_reward"] = weighted_style_reward.mean().detach()
    log.update(amp_metrics)
    return obs, reward, dones, extras

  def close(self) -> None:
    self.env.close()


class AMPVelocityOnPolicyRunner(VelocityOnPolicyRunner):
  env: AMPRewardVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict[str, Any],
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    if not isinstance(env, RslRlVecEnvWrapper):
      raise TypeError("AMPVelocityOnPolicyRunner expects RslRlVecEnvWrapper")
    amp_cfg = env.unwrapped.cfg.amp_style
    if not isinstance(amp_cfg, AmpStyleCfg):
      raise TypeError("AMP task requires env.cfg.amp_style = AmpStyleCfg")
    super().__init__(AMPRewardVecEnvWrapper(env, amp_cfg), train_cfg, log_dir, device)

  def save(self, path: str, infos=None):
    amp_state = self.env.amp_style.state_dict()
    infos = {**(infos or {}), "amp_state": amp_state}
    super().save(path, infos)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    infos = super().load(path, load_cfg, strict, map_location)
    if infos and "amp_state" in infos:
      self.env.amp_style.load_state_dict(infos["amp_state"])
    return infos
