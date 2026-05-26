import numpy as np
import torch

from mjlab.tasks.amp.motion import AMPMotionLibrary

BODY_NAMES = [
  "pelvis",
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


def test_amp_motion_samples_frame_bucket_matching_command(tmp_path) -> None:
  motion_path = tmp_path / "motion.npz"
  frame_commands = np.array(
    [
      [0.01, 0.0, 0.0],
      [0.31, 0.0, 0.0],
      [0.61, 0.0, 0.0],
      [-0.31, 0.0, 0.0],
    ],
    dtype=np.float32,
  )
  root_linear_velocities_base = np.column_stack(
    [frame_commands[:, :2], np.zeros(frame_commands.shape[0], dtype=np.float32)]
  )
  body_rotations = np.zeros((4, len(BODY_NAMES), 4), dtype=np.float32)
  body_rotations[..., 0] = 1.0
  np.savez(
    motion_path,
    fps=np.array(50.0, dtype=np.float32),
    dof_names=np.array([f"joint_{idx}" for idx in range(29)]),
    body_names=np.array(BODY_NAMES),
    dof_positions=np.zeros((4, 29), dtype=np.float32),
    dof_velocities=np.zeros((4, 29), dtype=np.float32),
    body_positions=np.zeros((4, len(BODY_NAMES), 3), dtype=np.float32),
    body_rotations=body_rotations,
    body_linear_velocities=np.zeros((4, len(BODY_NAMES), 3), dtype=np.float32),
    body_angular_velocities=np.zeros((4, len(BODY_NAMES), 3), dtype=np.float32),
    root_linear_velocities_base=root_linear_velocities_base,
    root_yaw_rates=frame_commands[:, 2],
    motion_command=np.zeros(3, dtype=np.float32),
  )

  library = AMPMotionLibrary(str(motion_path), device="cpu")
  commands = torch.tensor([[0.32, 0.0, 0.0], [0.62, 0.0, 0.0], [-0.32, 0.0, 0.0]])

  library.sample_reference(num_samples=3, history=1, frame_dt=0.02, commands=commands)

  expected = torch.tensor([[0.31, 0.0, 0.0], [0.61, 0.0, 0.0], [-0.31, 0.0, 0.0]])
  torch.testing.assert_close(library.last_sampled_commands, expected)
