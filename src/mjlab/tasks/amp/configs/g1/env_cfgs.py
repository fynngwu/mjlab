import os
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.amp.config import AmpStyleCfg
from mjlab.tasks.velocity import mdp
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
  relative_path = Path("holosoma/holosoma/data/motions/g1_29dof/amp/walk_and_run")
  repo_root = Path(__file__).resolve().parents[6]
  candidates = (
    os.environ.get("MJLAB_AMP_MOTION_PATH"),
    repo_root / "data/motions/g1_29dof/amp/walk_and_run",
    repo_root.parent / "src" / relative_path,
    repo_root.parent / "holosoma/src" / relative_path,
  )
  for candidate in candidates:
    if candidate is not None and Path(candidate).exists():
      return str(candidate)
  return str(repo_root / "data/motions/g1_29dof/amp/walk_and_run")


def unitree_g1_flat_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_env_cfg(play=play)
  cfg.scene.num_envs = 4096
  cfg.amp_style = AmpStyleCfg(
    motion_path=_amp_motion_path(),
  )

  for reward_name in _AMP_STYLE_REWARDS:
    cfg.rewards.pop(reward_name, None)

  cfg.terminations["base_height"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={"minimum_height": 0.2},
  )
  cfg.rewards["termination"] = RewardTermCfg(
    func=mdp.is_terminated,
    weight=-50.0,
    params={},
  )

  return cfg
