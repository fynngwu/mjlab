from __future__ import annotations

import os
import time
from typing import Any

import torch
from rsl_rl.env import VecEnv
from rsl_rl.utils import check_nan
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
    self.amp_style.store_policy_observations(amp_obs)
    weighted_style_reward = self.amp_style.cfg.reward_weight * style_reward
    reward = task_reward + weighted_style_reward

    log = extras.setdefault("log", {})
    log["AMP/task_reward"] = task_reward.mean().detach()
    log["AMP/style_reward"] = style_reward.mean().detach()
    log["AMP/weighted_style_reward"] = weighted_style_reward.mean().detach()
    return obs, reward, dones, extras

  def update_amp_discriminator(self, num_updates: int) -> dict[str, torch.Tensor]:
    return self.amp_style.update_many(num_updates)

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

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()

    if self.is_distributed:
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()

    self.logger.init_logging_writer()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      start = time.time()
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          actions = self.alg.act(obs)
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          if self.cfg.get("check_for_nan", True):
            check_nan(obs, rewards, dones)
          obs, rewards, dones = (
            obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
          )
          self.alg.process_env_step(obs, rewards, dones, extras)
          intrinsic_rewards = (
            self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
          )
          self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

        stop = time.time()
        collect_time = stop - start
        start = stop
        self.alg.compute_returns(obs)

      amp_metrics = self.env.update_amp_discriminator(self._amp_update_count())
      loss_dict = self.alg.update()
      loss_dict.update({key: value.item() for key, value in amp_metrics.items()})

      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it

      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
      )
      self._log_amp_metrics(it, amp_metrics)

      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

    if self.logger.writer is not None:
      self.save(
        os.path.join(
          self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"
        )
      )  # type: ignore[arg-type]
      self.logger.stop_logging_writer()

  def _amp_update_count(self) -> int:
    return int(self.alg.num_learning_epochs * self.alg.num_mini_batches)

  def _log_amp_metrics(
    self, iteration: int, metrics: dict[str, torch.Tensor]
  ) -> None:
    if self.logger.writer is None:
      return
    for key, value in metrics.items():
      self.logger.writer.add_scalar(key, value.item(), iteration)

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
