from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from beyondAMP.motion.motion_dataset import MotionDataset

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    
    
def reset_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

def reset_joints_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids, asset_cfg.joint_ids].clone()

    # bias these values randomly
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(
        joint_pos.view(len(env_ids), -1),
        joint_vel.view(len(env_ids), -1),
        env_ids=env_ids,
        joint_ids=asset_cfg.joint_ids,
    )


def reset_to_ref_motion_dataset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    joint_position_range: tuple[float, float],   # OFFSET noise, not scale
    joint_velocity_range: tuple[float, float],   # OFFSET noise
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Reset env states using motion dataset.
    Additive noise for joints (offset), not multiplicative scaling.
    """
    if env_ids.numel() < 1:
        return

    # ---------------------------------------------------------
    # Fallback: no dataset
    # ---------------------------------------------------------
    if getattr(env, "motion_dataset", None) is None:
        reset_root_state_uniform(env, env_ids, pose_range, velocity_range, asset_cfg)
        reset_joints_by_offset(env, env_ids, joint_position_range, joint_velocity_range, asset_cfg)
        return

    # ---------------------------------------------------------
    # Load motion data batch
    # ---------------------------------------------------------
    motion_dataset: MotionDataset = env.motion_dataset
    batch_size = env_ids.numel()

    ids, _ = motion_dataset.sample_batch(batch_size)

    joint_pos = motion_dataset.joint_pos[ids].clone()      # [B, J]
    joint_vel = motion_dataset.joint_vel[ids].clone()      # [B, J]
    anchor_pos = motion_dataset.anchor_pos_w[ids]          # [B, 3]
    anchor_quat = motion_dataset.anchor_quat_w[ids]        # [B, 4]
    anchor_lin_vel = motion_dataset.anchor_lin_vel_w[ids]  # [B, 3]
    anchor_ang_vel = motion_dataset.anchor_ang_vel_w[ids]  # [B, 3]

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    device = asset.device

    # ---------------------------------------------------------
    # Root pose noise
    # ---------------------------------------------------------
    pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    pose_ranges = torch.tensor([pose_range.get(k, (0.0, 0.0)) for k in pose_keys], device=device)
    pose_noise = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (batch_size, 6), device
    )

    positions = anchor_pos + env.scene.env_origins[env_ids] + pose_noise[:, :3]

    roll_noise, pitch_noise, yaw_noise = pose_noise[:, 3], pose_noise[:, 4], pose_noise[:, 5]
    rot_noise = math_utils.quat_from_euler_xyz(roll_noise, pitch_noise, yaw_noise)
    orientations = math_utils.quat_mul(rot_noise, anchor_quat)

    # ---------------------------------------------------------
    # Root velocity noise
    # ---------------------------------------------------------
    vel_keys = ["vx", "vy", "vz", "wx", "wy", "wz"]
    vel_ranges = torch.tensor([velocity_range.get(k, (0.0, 0.0)) for k in vel_keys], device=device)
    vel_noise = math_utils.sample_uniform(
        vel_ranges[:, 0], vel_ranges[:, 1], (batch_size, 6), device
    )

    lin_vel = anchor_lin_vel + vel_noise[:, :3]
    ang_vel = anchor_ang_vel + vel_noise[:, 3:6]

    # ---------------------------------------------------------
    # Joint additive noise (offset)
    # ---------------------------------------------------------
    # Example dict:
    # joint_position_range = {"low": -0.05, "high": 0.05}
    jp_low, jp_high = joint_position_range[0], joint_position_range[1]
    jv_low, jv_high = joint_velocity_range[0], joint_velocity_range[1]

    joint_pos_noise = math_utils.sample_uniform(
        torch.tensor(jp_low, device=device),
        torch.tensor(jp_high, device=device),
        joint_pos.shape,
        device,
    )
    joint_vel_noise = math_utils.sample_uniform(
        torch.tensor(jv_low, device=device),
        torch.tensor(jv_high, device=device),
        joint_vel.shape,
        device,
    )

    joint_pos = joint_pos + joint_pos_noise
    joint_vel = joint_vel + joint_vel_noise

    # clamp by limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids, asset_cfg.joint_ids]
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids, asset_cfg.joint_ids]

    joint_pos = joint_pos.clamp(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel = joint_vel.clamp(-joint_vel_limits, joint_vel_limits)

    # ---------------------------------------------------------
    # Write to simulator
    # ---------------------------------------------------------
    root_pose = torch.cat([positions, orientations], dim=-1)
    root_vel = torch.cat([lin_vel, ang_vel], dim=-1)

    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids)

    asset.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        env_ids=env_ids,
        joint_ids=asset_cfg.joint_ids,
    )

def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)