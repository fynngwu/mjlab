from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn, optim

from mjlab.entity import Entity
from mjlab.tasks.amp.config import AmpStyleCfg
from mjlab.tasks.amp.motion import AMP_KEY_BODY_NAMES, AMPMotionLibrary

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

SIM_BODY_ALIASES = {
  "left_rubber_hand": "left_wrist_yaw_link",
  "right_rubber_hand": "right_wrist_yaw_link",
}


def quat_apply_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  shape = vec.shape
  quat = quat.reshape(-1, 4)
  vec = vec.reshape(-1, 3)
  xyz = quat[:, 1:]
  w = quat[:, :1]
  cross = xyz.cross(vec, dim=-1) * 2
  return (vec + w * cross + xyz.cross(cross, dim=-1)).view(shape)


def quaternion_to_tangent_and_normal(quat_wxyz: torch.Tensor) -> torch.Tensor:
  tangent_ref = torch.zeros_like(quat_wxyz[..., :3])
  normal_ref = torch.zeros_like(quat_wxyz[..., :3])
  tangent_ref[..., 0] = 1
  normal_ref[..., -1] = 1
  tangent = quat_apply_wxyz(quat_wxyz, tangent_ref)
  normal = quat_apply_wxyz(quat_wxyz, normal_ref)
  return torch.cat([tangent, normal], dim=-1)


def compute_amp_frame(
  dof_pos: torch.Tensor,
  dof_vel: torch.Tensor,
  root_pos: torch.Tensor,
  root_quat_wxyz: torch.Tensor,
  root_lin_vel: torch.Tensor,
  root_ang_vel: torch.Tensor,
  key_body_pos: torch.Tensor,
) -> torch.Tensor:
  return torch.cat(
    (
      dof_pos,
      dof_vel,
      root_pos[:, 2:3],
      quaternion_to_tangent_and_normal(root_quat_wxyz),
      root_lin_vel,
      root_ang_vel,
      (key_body_pos - root_pos.unsqueeze(-2)).flatten(start_dim=1),
    ),
    dim=-1,
  )


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


class AMPStyleReward:
  def __init__(
    self, env: ManagerBasedRlEnv, cfg: AmpStyleCfg, device: str | torch.device
  ):
    self.env = env
    self.cfg = cfg
    self.device = device
    self.frame_dim = 29 + 29 + 1 + 6 + 3 + 3 + len(AMP_KEY_BODY_NAMES) * 3
    self.history = cfg.history
    self.obs_dim = self.frame_dim * self.history
    self.motion = AMPMotionLibrary(cfg.motion_path, device=device)
    self.asset: Entity = env.scene["robot"]
    self.body_indexes = self._body_indexes(AMP_KEY_BODY_NAMES)
    self.ref_body_index = self._body_indexes(["pelvis"])[0]
    self.dof_indexes = self.motion.get_dof_index(list(self.asset.joint_names))
    if self.frame_dim != 101:
      raise ValueError(
        f"humanoid_amp G1 AMP frame dim must be 101, got {self.frame_dim}"
      )
    self.frame_dt = float(env.step_dt)
    if abs(self.frame_dt - self.motion.dt) > 1.0e-4:
      raise ValueError(
        f"AMP motion dt {self.motion.dt} must match env dt {self.frame_dt}"
      )
    self.obs_buffer = torch.zeros(
      (env.num_envs, self.history, self.frame_dim), device=device
    )
    self.discriminator = AMPDiscriminator(self.obs_dim).to(device)
    self.optimizer = optim.AdamW(
      self.discriminator.parameters(),
      lr=cfg.learning_rate,
      weight_decay=cfg.weight_decay,
    )
    self.replay = AMPReplayBuffer(cfg.replay_size, self.obs_dim, device)

  def state_dict(self) -> dict[str, object]:
    return {
      "discriminator": self.discriminator.state_dict(),
      "optimizer": self.optimizer.state_dict(),
    }

  def load_state_dict(self, state: dict[str, object]) -> None:
    self.discriminator.load_state_dict(state["discriminator"])
    self.optimizer.load_state_dict(state["optimizer"])

  def _body_indexes(self, body_names: list[str]) -> list[int]:
    sim_body_names = list(self.asset.body_names)
    indexes = []
    for name in body_names:
      sim_name = SIM_BODY_ALIASES.get(name, name)
      if sim_name not in sim_body_names:
        raise ValueError(
          f"AMP body '{name}' maps to missing sim body '{sim_name}'. "
          f"Available bodies: {sim_body_names}"
        )
      indexes.append(sim_body_names.index(sim_name))
    return indexes

  @torch.no_grad()
  def reset(self) -> torch.Tensor:
    frame = self._compute_sim_frame()
    self.obs_buffer[:] = frame[:, None, :]
    return self.flattened_obs()

  @torch.no_grad()
  def observe(self, dones: torch.Tensor | None = None) -> torch.Tensor:
    frame = self._compute_sim_frame()
    self.obs_buffer[:, 1:] = self.obs_buffer[:, :-1].clone()
    self.obs_buffer[:, 0] = frame
    if dones is not None and dones.any():
      self.obs_buffer[dones.bool()] = frame[dones.bool(), None, :]
    return self.flattened_obs()

  def flattened_obs(self) -> torch.Tensor:
    return self.obs_buffer.reshape(self.env.num_envs, self.obs_dim)

  @torch.no_grad()
  def compute_reward(self, amp_obs: torch.Tensor) -> torch.Tensor:
    logits = self.discriminator(amp_obs)
    return (
      -torch.log(torch.clamp(1.0 - torch.sigmoid(logits), min=1.0e-4))
      * self.cfg.reward_scale
    )

  def update(self, policy_amp_obs: torch.Tensor) -> dict[str, torch.Tensor]:
    commands = self._commands()
    self.replay.add(policy_amp_obs, commands)
    if self.replay.size < self.cfg.batch_size:
      zero = torch.tensor(0.0, device=self.device)
      return self._metrics(zero, zero, zero, zero)

    with torch.inference_mode(False), torch.enable_grad():
      policy_obs, policy_commands = self.replay.sample(self.cfg.batch_size)
      demo_obs = self.collect_reference_motions(self.cfg.batch_size, policy_commands)
      demo_logits = self.discriminator(demo_obs)
      policy_logits = self.discriminator(policy_obs)
      demo_loss = F.binary_cross_entropy_with_logits(
        demo_logits, torch.ones_like(demo_logits)
      )
      policy_loss = F.binary_cross_entropy_with_logits(
        policy_logits, torch.zeros_like(policy_logits)
      )
      reg_loss = self.cfg.logit_reg * (
        demo_logits.square().mean() + policy_logits.square().mean()
      )
      grad_penalty = self.cfg.grad_penalty * self._gradient_penalty(demo_obs)
      loss = demo_loss + policy_loss + reg_loss + grad_penalty

      self.optimizer.zero_grad(set_to_none=True)
      loss.backward()
      self.optimizer.step()
    return self._metrics(
      loss.detach(),
      torch.sigmoid(demo_logits.detach()).mean(),
      torch.sigmoid(policy_logits.detach()).mean(),
      self._demo_command_error(),
    )

  def collect_reference_motions(
    self, num_samples: int, commands: torch.Tensor | None = None
  ) -> torch.Tensor:
    dof_pos, dof_vel, body_pos, body_quat, body_lin_vel, body_ang_vel = (
      self.motion.sample_reference(num_samples, self.history, self.frame_dt, commands)
    )
    frames = compute_amp_frame(
      dof_pos[:, self.dof_indexes],
      dof_vel[:, self.dof_indexes],
      body_pos[:, self.motion.ref_body_index],
      body_quat[:, self.motion.ref_body_index],
      body_lin_vel[:, self.motion.ref_body_index],
      body_ang_vel[:, self.motion.ref_body_index],
      body_pos[:, self.motion.key_body_indexes],
    )
    return frames.view(num_samples, self.history, self.frame_dim).reshape(
      num_samples, self.obs_dim
    )

  def _commands(self) -> torch.Tensor:
    command = self.env.command_manager.get_command("twist")
    if command is None:
      return torch.zeros((self.env.num_envs, 3), device=self.device)
    return command[:, :3]

  def _demo_command_error(self) -> torch.Tensor:
    if self.motion.last_sampled_commands.numel() == 0:
      return torch.tensor(0.0, device=self.device)
    error = self.motion.last_sampled_commands - self.motion.last_target_commands
    return torch.linalg.norm(error, dim=-1).mean()

  def _compute_sim_frame(self) -> torch.Tensor:
    data = self.asset.data
    return compute_amp_frame(
      data.joint_pos[:, self.dof_indexes],
      data.joint_vel[:, self.dof_indexes],
      data.body_link_pos_w[:, self.ref_body_index],
      data.body_link_quat_w[:, self.ref_body_index],
      data.body_link_lin_vel_w[:, self.ref_body_index],
      data.body_link_ang_vel_w[:, self.ref_body_index],
      data.body_link_pos_w[:, self.body_indexes],
    )

  def _gradient_penalty(self, demo_obs: torch.Tensor) -> torch.Tensor:
    demo_obs = demo_obs.detach().requires_grad_(True)
    logits = self.discriminator(demo_obs)
    grad = torch.autograd.grad(logits.sum(), demo_obs, create_graph=True)[0]
    return grad.square().sum(dim=-1).mean()

  def _metrics(
    self, loss, demo_score, policy_score, command_error
  ) -> dict[str, torch.Tensor]:
    return {
      "AMP/disc_loss": loss,
      "AMP/demo_score": demo_score,
      "AMP/policy_score": policy_score,
      "AMP/demo_command_error": command_error,
    }
