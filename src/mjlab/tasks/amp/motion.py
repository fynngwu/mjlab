from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

AMP_KEY_BODY_NAMES = [
  "left_shoulder_pitch_link",
  "right_shoulder_pitch_link",
  "left_elbow_link",
  "right_elbow_link",
  "right_hip_yaw_link",
  "left_hip_yaw_link",
  "right_rubber_hand",
  "left_rubber_hand",
  "right_ankle_roll_link",
  "left_ankle_roll_link",
]


@dataclass
class AMPMotion:
  name: str
  fps: float
  dof_names: list[str]
  body_names: list[str]
  command: torch.Tensor
  dof_pos: torch.Tensor
  dof_vel: torch.Tensor
  body_pos: torch.Tensor
  body_quat_wxyz: torch.Tensor
  body_lin_vel: torch.Tensor
  body_ang_vel: torch.Tensor

  @property
  def dt(self) -> float:
    return 1.0 / self.fps

  @property
  def duration(self) -> float:
    return max((self.dof_pos.shape[0] - 1) * self.dt, self.dt)


class AMPMotionLibrary:
  def __init__(self, motion_path: str, device: str | torch.device):
    self.device = device
    paths = self._resolve_motion_paths(motion_path)
    self.motions = [self._load_motion(path) for path in paths]
    self.dof_names = self.motions[0].dof_names
    self.body_names = self.motions[0].body_names
    self._validate_motion_contract()
    self.key_body_indexes = self.get_body_index(AMP_KEY_BODY_NAMES)
    self.ref_body_index = self.get_body_index(["pelvis"])[0]
    self.dt = 1.0 / self.motions[0].fps
    self.motion_commands = torch.stack([m.command for m in self.motions]).to(device)
    self.last_sampled_commands = torch.empty((0, 3), device=device)
    self.last_target_commands = torch.empty((0, 3), device=device)

  def _resolve_motion_paths(self, motion_path: str) -> list[Path]:
    raw_path = Path(motion_path)
    candidates = [raw_path]
    for parent in Path(__file__).resolve().parents:
      candidates.append(parent / raw_path)
    path = next((candidate for candidate in candidates if candidate.exists()), raw_path)
    paths = sorted(path.rglob("*.npz")) if path.is_dir() else [path]
    if not paths or not all(path.exists() for path in paths):
      raise FileNotFoundError(f"No AMP motion npz files found at {motion_path}")
    return paths

  def _load_motion(self, path: Path) -> AMPMotion:
    data = np.load(path, allow_pickle=True)
    command = data["motion_command"] if "motion_command" in data else np.zeros(3)
    return AMPMotion(
      name=path.name,
      fps=float(np.asarray(data["fps"]).reshape(-1)[0]),
      dof_names=[str(name) for name in data["dof_names"]],
      body_names=[str(name) for name in data["body_names"]],
      command=torch.as_tensor(command, device=self.device, dtype=torch.float),
      dof_pos=torch.as_tensor(data["dof_positions"], device=self.device).float(),
      dof_vel=torch.as_tensor(data["dof_velocities"], device=self.device).float(),
      body_pos=torch.as_tensor(data["body_positions"], device=self.device).float(),
      body_quat_wxyz=torch.as_tensor(
        data["body_rotations"], device=self.device
      ).float(),
      body_lin_vel=torch.as_tensor(
        data["body_linear_velocities"], device=self.device
      ).float(),
      body_ang_vel=torch.as_tensor(
        data["body_angular_velocities"], device=self.device
      ).float(),
    )

  def _validate_motion_contract(self) -> None:
    expected_bodies = ["pelvis", *AMP_KEY_BODY_NAMES]
    if self.body_names != expected_bodies:
      raise ValueError(
        f"AMP body_names must be {expected_bodies}, got {self.body_names}"
      )
    if abs(self.motions[0].fps - 50.0) > 1.0e-3:
      raise ValueError(f"AMP motion fps must be 50Hz, got {self.motions[0].fps}")
    for motion in self.motions[1:]:
      if motion.dof_names != self.dof_names:
        raise ValueError(f"{motion.name}: AMP dof_names differ")
      if motion.body_names != self.body_names:
        raise ValueError(f"{motion.name}: AMP body_names differ")
      if abs(motion.fps - self.motions[0].fps) > 1.0e-3:
        raise ValueError(f"{motion.name}: AMP fps differs")

  def get_dof_index(self, dof_names: list[str]) -> list[int]:
    return [self.dof_names.index(name) for name in dof_names]

  def get_body_index(self, body_names: list[str]) -> list[int]:
    return [self.body_names.index(name) for name in body_names]

  def sample_reference(
    self,
    num_samples: int,
    history: int,
    frame_dt: float,
    commands: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, ...]:
    motion_ids = self._sample_motion_ids(num_samples, commands)
    current_times = torch.empty(num_samples, device=self.device)
    for motion_id, motion in enumerate(self.motions):
      mask = motion_ids == motion_id
      if mask.any():
        current_times[mask] = (
          torch.rand(mask.sum(), device=self.device) * motion.duration
        )
    self.last_sampled_commands = self.motion_commands[motion_ids]
    self.last_target_commands = (
      torch.zeros_like(self.last_sampled_commands)
      if commands is None
      else commands[:, :3]
    )
    offsets = torch.arange(history, device=self.device, dtype=torch.float) * frame_dt
    return self.sample(
      motion_ids.repeat_interleave(history),
      (current_times[:, None] - offsets).flatten(),
    )

  def _sample_motion_ids(
    self, num_samples: int, commands: torch.Tensor | None
  ) -> torch.Tensor:
    if commands is None:
      return torch.randint(len(self.motions), (num_samples,), device=self.device)
    target = commands[:, :3].to(device=self.device, dtype=torch.float)
    scale = torch.tensor([0.5, 0.5, 0.75], device=self.device)
    diff = (target[:, None, :] - self.motion_commands[None, :, :]) / scale
    weights = torch.softmax(-diff.square().sum(dim=-1), dim=-1)
    return torch.multinomial(weights, 1).squeeze(1)

  def sample(
    self, motion_ids: torch.Tensor, times: torch.Tensor
  ) -> tuple[torch.Tensor, ...]:
    chunks = [[] for _ in range(6)]
    for motion_id, motion in enumerate(self.motions):
      mask = motion_ids == motion_id
      if mask.any():
        sampled = self._sample_single(motion, times[mask])
        for idx, value in enumerate(sampled):
          chunks[idx].append((mask, value))
    return tuple(self._merge(len(times), chunk) for chunk in chunks)

  def _sample_single(
    self, motion: AMPMotion, times: torch.Tensor
  ) -> tuple[torch.Tensor, ...]:
    times = torch.remainder(times, motion.duration)
    frame = times / motion.dt
    index0 = torch.floor(frame).long().clamp(max=motion.dof_pos.shape[0] - 1)
    index1 = (index0 + 1).clamp(max=motion.dof_pos.shape[0] - 1)
    alpha = (frame - index0.float()).view(-1, 1)
    body_alpha = alpha.view(-1, 1, 1)
    return (
      self._lerp(motion.dof_pos[index0], motion.dof_pos[index1], alpha),
      self._lerp(motion.dof_vel[index0], motion.dof_vel[index1], alpha),
      self._lerp(motion.body_pos[index0], motion.body_pos[index1], body_alpha),
      torch.nn.functional.normalize(
        self._lerp(
          motion.body_quat_wxyz[index0], motion.body_quat_wxyz[index1], body_alpha
        ),
        dim=-1,
      ),
      self._lerp(motion.body_lin_vel[index0], motion.body_lin_vel[index1], body_alpha),
      self._lerp(motion.body_ang_vel[index0], motion.body_ang_vel[index1], body_alpha),
    )

  def _merge(
    self, num_samples: int, chunks: list[tuple[torch.Tensor, torch.Tensor]]
  ) -> torch.Tensor:
    shape = (num_samples,) + chunks[0][1].shape[1:]
    merged = torch.empty(shape, device=self.device, dtype=torch.float)
    for mask, values in chunks:
      merged[mask] = values
    return merged

  @staticmethod
  def _lerp(
    left: torch.Tensor, right: torch.Tensor, alpha: torch.Tensor
  ) -> torch.Tensor:
    return left + (right - left) * alpha
