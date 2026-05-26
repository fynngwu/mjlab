from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.amp.config import AmpStyleCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

_AMP_STYLE_REWARDS = (
  "pose",
  "air_time",
  "foot_clearance",
  "foot_swing_height",
  "foot_slip",
  "soft_landing",
  "angular_momentum",
  "body_ang_vel",
)


def _amp_motion_path() -> str:
  repo_root = Path(__file__).resolve().parents[7]
  return str(repo_root / "src/holosoma/holosoma/data/motions/g1_29dof/amp/walk_and_run")


def unitree_g1_flat_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_env_cfg(play=play)
  cfg.scene.num_envs = 4096
  cfg.amp_style = AmpStyleCfg(
    motion_path=_amp_motion_path(),
    reward_weight=5.0,
  )

  for reward_name in _AMP_STYLE_REWARDS:
    cfg.rewards.pop(reward_name, None)

  return cfg
