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

COMMAND_BUCKET_RESOLUTION = 0.3


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
  frame_commands: torch.Tensor

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
    self._build_command_buckets()
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
    frame_commands = self._load_frame_commands(data, command)
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
      frame_commands=torch.as_tensor(
        frame_commands, device=self.device, dtype=torch.float
      ),
    )

  def _load_frame_commands(
    self, data: np.lib.npyio.NpzFile, fallback_command: np.ndarray
  ) -> np.ndarray:
    if "root_linear_velocities_base" not in data or "root_yaw_rates" not in data:
      frame_count = data["dof_positions"].shape[0]
      command = np.asarray(fallback_command, dtype=np.float32)[None]
      return np.repeat(command, frame_count, axis=0)
    root_lin_vel_b = np.asarray(data["root_linear_velocities_base"], dtype=np.float32)
    root_yaw_rate = np.asarray(data["root_yaw_rates"], dtype=np.float32).reshape(-1, 1)
    return np.concatenate((root_lin_vel_b[:, :2], root_yaw_rate), axis=1)

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
      if motion.frame_commands.shape != (motion.dof_pos.shape[0], 3):
        raise ValueError(
          f"{motion.name}: AMP frame_commands must have shape "
          f"({motion.dof_pos.shape[0]}, 3), got {motion.frame_commands.shape}"
        )

  def get_dof_index(self, dof_names: list[str]) -> list[int]:
    return [self.dof_names.index(name) for name in dof_names]

  def get_body_index(self, body_names: list[str]) -> list[int]:
    return [self.body_names.index(name) for name in body_names]

  def _build_command_buckets(self) -> None:
    motion_ids = []
    frame_ids = []
    frame_commands = []
    for motion_id, motion in enumerate(self.motions):
      num_frames = motion.dof_pos.shape[0]
      motion_ids.append(
        torch.full((num_frames,), motion_id, device=self.device, dtype=torch.long)
      )
      frame_ids.append(torch.arange(num_frames, device=self.device, dtype=torch.long))
      frame_commands.append(motion.frame_commands)
    self._frame_motion_ids = torch.cat(motion_ids)
    self._frame_ids = torch.cat(frame_ids)
    self._frame_commands = torch.cat(frame_commands)

    bucket_ids = self._bucketize_commands(self._frame_commands).cpu().tolist()
    bucket_to_indexes: dict[tuple[int, int, int], list[int]] = {}
    for index, bucket in enumerate(bucket_ids):
      bucket_to_indexes.setdefault(tuple(bucket), []).append(index)
    self._bucket_keys = torch.tensor(
      list(bucket_to_indexes), device=self.device, dtype=torch.long
    )
    self._bucket_frame_indexes = [
      torch.tensor(indexes, device=self.device, dtype=torch.long)
      for indexes in bucket_to_indexes.values()
    ]

  def _bucketize_commands(self, commands: torch.Tensor) -> torch.Tensor:
    return torch.round(commands[:, :3] / COMMAND_BUCKET_RESOLUTION).long()

  def sample_reference(
    self,
    num_samples: int,
    history: int,
    frame_dt: float,
    commands: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, ...]:
    motion_ids, frame_ids, frame_commands = self._sample_frame_refs(
      num_samples, commands
    )
    current_times = frame_ids.float() * self.dt
    self.last_sampled_commands = frame_commands
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

  def _sample_frame_refs(
    self, num_samples: int, commands: torch.Tensor | None
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if commands is None:
      frame_indexes = torch.randint(
        self._frame_ids.numel(), (num_samples,), device=self.device
      )
      return (
        self._frame_motion_ids[frame_indexes],
        self._frame_ids[frame_indexes],
        self._frame_commands[frame_indexes],
      )
    target = commands[:, :3].to(device=self.device, dtype=torch.float)
    bucket_indexes = self._resolve_bucket_indexes(self._bucketize_commands(target))
    frame_indexes = torch.empty(num_samples, device=self.device, dtype=torch.long)
    for bucket_index in torch.unique(bucket_indexes).tolist():
      mask = bucket_indexes == bucket_index
      candidates = self._bucket_frame_indexes[bucket_index]
      choices = torch.randint(
        candidates.numel(), (int(mask.sum()),), device=self.device
      )
      frame_indexes[mask] = candidates[choices]
    return (
      self._frame_motion_ids[frame_indexes],
      self._frame_ids[frame_indexes],
      self._frame_commands[frame_indexes],
    )

  def _resolve_bucket_indexes(self, target_buckets: torch.Tensor) -> torch.Tensor:
    matches = (target_buckets[:, None, :] == self._bucket_keys[None, :, :]).all(dim=-1)
    has_exact_match = matches.any(dim=-1)
    exact_indexes = matches.float().argmax(dim=-1)
    bucket_dist = (
      target_buckets[:, None, :] - self._bucket_keys[None, :, :]
    ).square().sum(dim=-1)
    nearest_indexes = bucket_dist.argmin(dim=-1)
    return torch.where(has_exact_match, exact_indexes, nearest_indexes)

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
