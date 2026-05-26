from __future__ import annotations

import os
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import check_nan, resolve_callable
from tensordict import TensorDict

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.amp.config import AmpStyleCfg
from mjlab.tasks.amp.style_reward import AMPStyleReward, gradient_penalty
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

    # Circular buffers for interleaved discriminator updates.
    self.disc_obs_buffer = CircularBuffer(
      max_len=cfg.disc_obs_buffer_size, batch_size=env.num_envs, device=env.device
    )
    self.disc_demo_obs_buffer = CircularBuffer(
      max_len=cfg.disc_obs_buffer_size, batch_size=env.num_envs, device=env.device
    )

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

    # Pre-compute demo observations for this step
    commands = self.amp_style._commands()
    demo_obs = self.amp_style.compute_demo_obs(commands)
    self.disc_obs_buffer.append(amp_obs)
    self.disc_demo_obs_buffer.append(demo_obs)

    style_reward = self.amp_style.compute_reward(amp_obs, dt=self.amp_style.frame_dt)
    weighted_style_reward = self.amp_style.cfg.reward_weight * style_reward
    reward = task_reward + weighted_style_reward

    log = extras.setdefault("log", {})
    log["AMP/task_reward"] = task_reward.mean().detach()
    log["AMP/style_reward"] = style_reward.mean().detach()
    log["AMP/weighted_style_reward"] = weighted_style_reward.mean().detach()
    return obs, reward, dones, extras

  def close(self) -> None:
    self.env.close()


class CircularBuffer:
  """Minimal circular buffer for stored batched tensor data.

  Mirrors the interface of ``rsl_rl.storage.CircularBuffer`` (from the forked
  rsl_rl used by legged_lab) so that ``PPOAMP`` can use the same mini-batch
  generator pattern.
  """

  def __init__(self, max_len: int, batch_size: int, device: str):
    self.max_len = max_len
    self.batch_size = batch_size
    self.device = device
    self._buffer: torch.Tensor | None = None
    self._ptr = 0
    self._size = 0

  def append(self, data: torch.Tensor) -> None:
    data = data.to(self.device)
    if self._buffer is None:
      self._buffer = torch.empty(
        (self.max_len, *data.shape), dtype=data.dtype, device=self.device
      )
    self._buffer[self._ptr] = data
    self._ptr = (self._ptr + 1) % self.max_len
    self._size = min(self._size + 1, self.max_len)

  def mini_batch_generator(
    self, fetch_length: int, num_mini_batches: int, num_epochs: int
  ):
    """Yield mini-batches of observations from the buffer.

    Each mini-batch has shape ``[mini_batch_size, *data_dims]``.
    """
    cur_len = min(self._size, self.max_len)
    if cur_len < fetch_length:
      raise RuntimeError(
        f"Buffer has {cur_len} entries, need at least {fetch_length}"
      )
    total = self.batch_size * fetch_length
    epoch_batch_size = total
    mini_batch_size = epoch_batch_size // num_mini_batches

    src_ptr = (self._ptr - cur_len) % self.max_len
    time_inds = torch.randint(cur_len, (epoch_batch_size,), device=self.device)
    buf_inds = (src_ptr + time_inds) % self.max_len
    env_inds = torch.randint(self.batch_size, (epoch_batch_size,), device=self.device)
    all_data = self._buffer[buf_inds, env_inds]

    for _ in range(num_epochs):
      perm = torch.randperm(epoch_batch_size, device=self.device)
      for i in range(num_mini_batches):
        start = i * mini_batch_size
        idx = perm[start : start + mini_batch_size]
        yield all_data[idx]


class PPOAMP(PPO):
  """PPO with interleaved AMP discriminator updates (mirrors legged_lab PPOAMP)."""

  def __init__(
    self,
    policy: Any,
    storage: Any,
    disc_obs_buffer: CircularBuffer,
    disc_demo_obs_buffer: CircularBuffer,
    amp_style: AMPStyleReward,
    amp_cfg: AmpStyleCfg,
    num_learning_epochs: int = 5,
    num_mini_batches: int = 4,
    clip_param: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    value_loss_coef: float = 1.0,
    entropy_coef: float = 0.01,
    learning_rate: float = 0.001,
    max_grad_norm: float = 1.0,
    use_clipped_value_loss: bool = True,
    schedule: str = "adaptive",
    desired_kl: float = 0.01,
    normalize_advantage_per_mini_batch: bool = False,
    device: str = "cpu",
    rnd_cfg: dict | None = None,
    symmetry_cfg: dict | None = None,
    multi_gpu_cfg: dict | None = None,
  ) -> None:
    super().__init__(
      policy,
      storage,
      num_learning_epochs,
      num_mini_batches,
      clip_param,
      gamma,
      lam,
      value_loss_coef,
      entropy_coef,
      learning_rate,
      max_grad_norm,
      use_clipped_value_loss,
      schedule,
      desired_kl,
      normalize_advantage_per_mini_batch,
      device,
      rnd_cfg,
      symmetry_cfg,
      multi_gpu_cfg,
    )

    self.amp_style = amp_style
    self.amp_cfg = amp_cfg
    self.disc_obs_buffer = disc_obs_buffer
    self.disc_demo_obs_buffer = disc_demo_obs_buffer

    self.disc_optimizer = optim.AdamW(
      amp_style.discriminator.parameters(),
      lr=amp_cfg.learning_rate,
      weight_decay=amp_cfg.weight_decay,
    )

  def process_env_step(
    self, obs: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
  ) -> None:
    """AMPRewardVecEnvWrapper.step() already computed style reward and appended disc buffers.
    Only need to call parent to store transition in RolloutStorage."""
    super().process_env_step(obs, rewards, dones, extras)

  def update(self) -> dict[str, float]:
    mean_value_loss = 0
    mean_surrogate_loss = 0
    mean_entropy = 0
    mean_rnd_loss = 0 if self.rnd else None
    mean_symmetry_loss = 0 if self.symmetry else None

    # AMP discriminator metrics
    mean_disc_loss = 0
    mean_disc_grad_penalty = 0
    mean_demo_score = 0
    mean_policy_score = 0

    # Get mini batch generator for PPO
    if self.policy.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    # Get mini batch generator for discriminator data
    fetch_length = self.storage.num_transitions_per_env
    disc_obs_generator = self.disc_obs_buffer.mini_batch_generator(
      fetch_length=fetch_length,
      num_mini_batches=self.num_mini_batches,
      num_epochs=self.num_learning_epochs,
    )
    disc_demo_obs_generator = self.disc_demo_obs_buffer.mini_batch_generator(
      fetch_length=fetch_length,
      num_mini_batches=self.num_mini_batches,
      num_epochs=self.num_learning_epochs,
    )

    # Iterate over batches — interleave PPO and discriminator updates
    for samples, disc_obs_batch, disc_demo_obs_batch in zip(
      generator, disc_obs_generator, disc_demo_obs_generator
    ):
      (
        obs_batch,
        actions_batch,
        target_values_batch,
        advantages_batch,
        returns_batch,
        old_actions_log_prob_batch,
        old_mu_batch,
        old_sigma_batch,
        hidden_states_batch,
        masks_batch,
      ) = samples

      num_aug = 1
      original_batch_size = obs_batch.batch_size[0]

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          advantages_batch = (advantages_batch - advantages_batch.mean()) / (
            advantages_batch.std() + 1e-8
          )

      # Symmetric augmentation
      if self.symmetry and self.symmetry["use_data_augmentation"]:
        data_augmentation_func = self.symmetry["data_augmentation_func"]
        obs_batch, actions_batch = data_augmentation_func(
          obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"]
        )
        num_aug = int(obs_batch.batch_size[0] / original_batch_size)
        old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
        target_values_batch = target_values_batch.repeat(num_aug, 1)
        advantages_batch = advantages_batch.repeat(num_aug, 1)
        returns_batch = returns_batch.repeat(num_aug, 1)

      # PPO forward
      self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
      actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
      value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
      mu_batch = self.policy.action_mean[:original_batch_size]
      sigma_batch = self.policy.action_std[:original_batch_size]
      entropy_batch = self.policy.entropy[:original_batch_size]

      # KL adaptation
      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = torch.sum(
            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
            / (2.0 * torch.square(sigma_batch))
            - 0.5,
            axis=-1,
          )
          kl_mean = torch.mean(kl)
          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()
          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      # PPO losses
      ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
      surrogate = -torch.squeeze(advantages_batch) * ratio
      surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (value_batch - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (returns_batch - value_batch).pow(2).mean()

      loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

      # Symmetry loss
      if self.symmetry:
        if not self.symmetry["use_data_augmentation"]:
          data_augmentation_func = self.symmetry["data_augmentation_func"]
          obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
          num_aug = int(obs_batch.shape[0] / original_batch_size)
        mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
        action_mean_orig = mean_actions_batch[:original_batch_size]
        _, actions_mean_symm_batch = data_augmentation_func(
          obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
        )
        mse_loss = torch.nn.MSELoss()
        symmetry_loss = mse_loss(
          mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
        )
        if self.symmetry["use_mirror_loss"]:
          loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
        else:
          symmetry_loss = symmetry_loss.detach()

      # RND loss
      if self.rnd:
        with torch.no_grad():
          rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
          rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
        predicted_embedding = self.rnd.predictor(rnd_state_batch)
        target_embedding = self.rnd.target(rnd_state_batch).detach()
        rnd_loss = torch.nn.MSELoss()(predicted_embedding, target_embedding)

      # AMP discriminator loss
      disc_score = self.amp_style.discriminator(disc_obs_batch)
      disc_demo_score = self.amp_style.discriminator(disc_demo_obs_batch)

      demo_loss = 0.5 * F.mse_loss(disc_demo_score, torch.ones_like(disc_demo_score))
      policy_loss = 0.5 * F.mse_loss(disc_score, -torch.ones_like(disc_score))
      reg_loss = self.amp_cfg.logit_reg * (
        disc_demo_score.square().mean() + disc_score.square().mean()
      )
      disc_grad_penalty_loss = self.amp_cfg.grad_penalty * gradient_penalty(
        self.amp_style.discriminator, disc_demo_obs_batch
      )
      disc_loss = self.amp_cfg.loss_scale * (demo_loss + policy_loss + reg_loss + disc_grad_penalty_loss)

      # Backward: PPO + RND + discriminator
      self.optimizer.zero_grad()
      loss.backward()

      if self.rnd:
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()

      self.disc_optimizer.zero_grad()
      disc_loss.backward()

      # Gradient clipping and stepping
      if self.is_multi_gpu:
        self.reduce_parameters()

      nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
      self.optimizer.step()

      if self.rnd_optimizer:
        self.rnd_optimizer.step()

      # NOTE: disc_max_grad_norm not exposed via AmpStyleCfg; skip clipping for now
      self.disc_optimizer.step()

      # Accumulate metrics
      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy_batch.mean().item()
      if mean_rnd_loss is not None:
        mean_rnd_loss += rnd_loss.item()
      if mean_symmetry_loss is not None:
        mean_symmetry_loss += symmetry_loss.item()
      mean_disc_loss += disc_loss.item()
      mean_disc_grad_penalty += disc_grad_penalty_loss.item()
      mean_demo_score += disc_demo_score.mean().item()
      mean_policy_score += disc_score.mean().item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    if mean_rnd_loss is not None:
      mean_rnd_loss /= num_updates
    if mean_symmetry_loss is not None:
      mean_symmetry_loss /= num_updates
    mean_disc_loss /= num_updates
    mean_disc_grad_penalty /= num_updates
    mean_demo_score /= num_updates
    mean_policy_score /= num_updates

    self.storage.clear()

    loss_dict = {
      "value": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
    }
    if self.rnd:
      loss_dict["rnd"] = mean_rnd_loss
    if self.symmetry:
      loss_dict["symmetry"] = mean_symmetry_loss
    loss_dict["AMP/disc_loss"] = mean_disc_loss
    loss_dict["AMP/disc_grad_penalty"] = mean_disc_grad_penalty
    loss_dict["AMP/demo_score"] = mean_demo_score
    loss_dict["AMP/policy_score"] = mean_policy_score

    return loss_dict


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

  def _construct_algorithm(self, obs: torch.Tensor) -> PPO:
    """Construct PPOAMP with discriminator buffers."""
    # Resolve RND / symmetry configs (mirrors OnPolicyRunner._construct_algorithm)
    self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
    self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

    # Resolve deprecated normalization config
    if self.cfg.get("empirical_normalization") is not None:
      import warnings
      warnings.warn(
        "The `empirical_normalization` parameter is deprecated.",
        DeprecationWarning,
      )
      if self.policy_cfg.get("actor_obs_normalization") is None:
        self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
      if self.policy_cfg.get("critic_obs_normalization") is None:
        self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

    # Policy
    actor_critic_class = resolve_callable(self.policy_cfg.pop("class_name"))
    actor_critic = actor_critic_class(
      obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
    ).to(self.device)

    # Storage
    storage = RolloutStorage(
      "rl", self.env.num_envs, self.cfg["num_steps_per_env"],
      obs, [self.env.num_actions], self.device,
    )

    # Create PPOAMP
    alg = PPOAMP(
      actor_critic,
      storage,
      disc_obs_buffer=self.env.disc_obs_buffer,
      disc_demo_obs_buffer=self.env.disc_demo_obs_buffer,
      amp_style=self.env.amp_style,
      amp_cfg=self.env.amp_style.cfg,
      device=self.device,
      **self.alg_cfg,
      multi_gpu_cfg=self.multi_gpu_cfg,
    )
    return alg

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

      # Single update call — PPOAMP interleaves discriminator inside
      loss_dict = self.alg.update()

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

      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

    if self.logger.writer is not None:
      self.save(
        os.path.join(
          self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"
        )
      )
      self.logger.stop_logging_writer()

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
