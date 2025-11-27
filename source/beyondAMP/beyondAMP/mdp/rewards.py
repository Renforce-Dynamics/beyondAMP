from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_exp_torso(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Track linear velocity in XY plane using yaw-only frame (ignores pitch/roll).
    
    This is suitable for crawling robots where the body is tilted but we still want
    to track velocity in the horizontal plane based on the robot's facing direction.
    
    Args:
        env: The training environment.
        command_name: The name of the command term to track.
        std: The standard deviation for the exponential reward.
        asset_cfg: SceneEntityCfg specifying the body to track (default: torso_link).
    
    Returns:
        Reward tensor based on tracking error.
    """
    # Get the command term
    command_term = env.command_manager.get_term(command_name)
    # Get the commanded velocity in XY plane
    command_vel = command_term.command[:, :2]
    
    # Get the actual velocity of the specified body
    asset: RigidObject = env.scene[asset_cfg.name]
    # Get body index using find_bodies
    body_idx = asset.find_bodies(asset_cfg.body_names, preserve_order=True)[0][0]
    body_lin_vel_w = asset.data.body_lin_vel_w[:, body_idx, :]
    
    # Use yaw-only quaternion from root (ignores pitch/roll)
    # This way "forward" is always in the horizontal plane
    root_quat_w = asset.data.root_quat_w
    yaw_quat = math_utils.yaw_quat(root_quat_w)
    
    # Transform world velocity to yaw-only frame
    body_lin_vel_yaw = math_utils.quat_rotate_inverse(yaw_quat, body_lin_vel_w)
    
    # Extract XY components (in horizontal plane)
    body_vel_xy = body_lin_vel_yaw[:, :2]
    
    # Compute tracking error
    error = torch.sum(torch.square(command_vel - body_vel_xy), dim=-1)
    
    # Return exponential reward
    return torch.exp(-error / std**2)


def track_ang_vel_z_exp_torso(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Track angular velocity around world Z axis (yaw rate) using exponential reward.
    
    This tracks rotation in the horizontal plane, suitable for crawling robots.
    
    Args:
        env: The training environment.
        command_name: The name of the command term to track.
        std: The standard deviation for the exponential reward.
        asset_cfg: SceneEntityCfg specifying the body to track (default: torso_link).
    
    Returns:
        Reward tensor based on tracking error.
    """
    # Get the command term
    command_term = env.command_manager.get_term(command_name)
    # Get the commanded angular velocity around Z axis
    command_ang_vel_z = command_term.command[:, 2]
    
    # Get the actual angular velocity of the specified body in world frame
    asset: RigidObject = env.scene[asset_cfg.name]
    body_idx = asset.find_bodies(asset_cfg.body_names, preserve_order=True)[0][0]
    body_ang_vel_w = asset.data.body_ang_vel_w[:, body_idx, :]
    
    # Directly use world frame Z component (vertical axis rotation)
    # This is the yaw rate regardless of body pitch/roll
    body_ang_vel_z = body_ang_vel_w[:, 2]
    
    # Compute tracking error
    error = torch.square(command_ang_vel_z - body_ang_vel_z)
    
    # Return exponential reward
    return torch.exp(-error / std**2)


def pelvis_upright(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="pelvis"),
) -> torch.Tensor:
    """Reward for keeping pelvis upright (vertical).
    
    The reward is based on how well the pelvis aligns with the vertical direction.
    When upright, gravity in pelvis body frame should point downward (z = -1).
    
    Args:
        env: The training environment.
        asset_cfg: SceneEntityCfg specifying the body to check (default: pelvis).
    
    Returns:
        Reward tensor (higher when pelvis is more upright).
    """
    # Get the asset
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Get pelvis body index
    body_idx = asset.find_bodies(asset_cfg.body_names, preserve_order=True)[0][0]
    
    # Get pelvis orientation in world frame
    pelvis_quat_w = asset.data.body_quat_w[:, body_idx, :]
    
    # Gravity vector in world frame (pointing downward)
    # Use GRAVITY_VEC_W if available, otherwise use [0, 0, -1]
    if hasattr(asset.data, 'GRAVITY_VEC_W'):
        gravity_w = asset.data.GRAVITY_VEC_W
        # Handle different shapes: [3], [1, num_envs, 3], or [num_envs, 3]
        if gravity_w.dim() == 1:
            # Shape [3], expand to [num_envs, 3]
            gravity_w = gravity_w.unsqueeze(0).expand(env.num_envs, -1)
        elif gravity_w.dim() == 3:
            # Shape [1, num_envs, 3], squeeze first dimension
            gravity_w = gravity_w.squeeze(0)
        # If dim() == 2, it's already [num_envs, 3], use as is
    else:
        # Create gravity vector [num_envs, 3]
        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    
    # Transform gravity to pelvis body frame
    gravity_b = math_utils.quat_rotate_inverse(pelvis_quat_w, gravity_w)
    
    # Reward is based on how close gravity aligns with -z axis in body frame
    # When upright, gravity should point in -z direction: [0, 0, -1]
    # Penalize both x (roll) and y (pitch) components
    # gravity_b[:, 0] and gravity_b[:, 1] should be close to 0
    # gravity_b[:, 2] should be close to -1
    
    # Penalize deviation in x and y directions (should be 0)
    xy_penalty = torch.square(gravity_b[:, 0]) + torch.square(gravity_b[:, 1])
    
    # Reward based on z alignment (should be -1)
    z_score = -gravity_b[:, 2]  # This should be close to 1 when upright
    
    # Combined reward: z_score penalized by xy deviation
    # When perfectly upright: xy_penalty=0, z_score=1, reward=1
    # When tilted: xy_penalty>0, reward decreases
    reward = torch.clamp(z_score - xy_penalty, min=0.0)
    
    return reward


def pelvis_forward_lean(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="pelvis"),
) -> torch.Tensor:
    """Reward for leaning pelvis forward by 90 degrees (horizontal).
    
    When leaning forward 90 degrees, gravity in body frame should point in +x direction.
    
    Args:
        env: The training environment.
        asset_cfg: SceneEntityCfg specifying the body to check (default: pelvis).
    
    Returns:
        Reward tensor (higher when pelvis is leaning forward 90 degrees).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    body_idx = asset.find_bodies(asset_cfg.body_names, preserve_order=True)[0][0]
    pelvis_quat_w = asset.data.body_quat_w[:, body_idx, :]
    
    if hasattr(asset.data, 'GRAVITY_VEC_W'):
        gravity_w = asset.data.GRAVITY_VEC_W
        if gravity_w.dim() == 1:
            gravity_w = gravity_w.unsqueeze(0).expand(env.num_envs, -1)
        elif gravity_w.dim() == 3:
            gravity_w = gravity_w.squeeze(0)
    else:
        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    
    gravity_b = math_utils.quat_rotate_inverse(pelvis_quat_w, gravity_w)
    
    # When leaning forward 90 degrees, gravity should point in +x direction: [1, 0, 0]
    # Reward based on x alignment (should be 1) and penalize y, z deviation
    x_score = gravity_b[:, 0]  # Should be close to 1 when leaning forward
    yz_penalty = torch.square(gravity_b[:, 1]) + torch.square(gravity_b[:, 2])
    
    reward = torch.clamp(x_score - yz_penalty, min=0.0)
    
    return reward


def single_limb_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    threshold: float = 0.5,
) -> torch.Tensor:
    """Reward for having at least one limb (knee/ankle) in the air.
    
    This encourages crawling gait where one limb is lifted while others 
    maintain contact with the ground.
    
    Args:
        env: The training environment.
        sensor_cfg: SceneEntityCfg specifying the contact sensor and body names.
        command_name: The name of the command term (only active when moving).
        threshold: Minimum air time to count as "in air".
    
    Returns:
        Reward tensor (1.0 when at least one limb is in air, 0.0 otherwise).
    """
    from isaaclab.sensors import ContactSensor
    
    # Get contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get air time for all specified bodies
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    
    # Check if at least one limb is in the air (air time > threshold)
    any_limb_in_air = torch.any(last_air_time > threshold, dim=-1).float()
    
    # Only reward when there's a velocity command (moving)
    command_term = env.command_manager.get_term(command_name)
    command_norm = torch.norm(command_term.command, dim=1)
    is_moving = (command_norm > 0.1).float()
    
    # Reward is 1.0 when moving and at least one limb is in air
    reward = any_limb_in_air * is_moving
    
    return reward


def diagonal_limbs_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    threshold: float = 0.02,
) -> torch.Tensor:
    """Reward for diagonal limbs (hand and foot) being in the air simultaneously.
    
    Encourages trot-like gait where diagonal pairs (left_hand + right_foot, 
    right_hand + left_foot) lift off together.
    
    Args:
        env: The training environment.
        sensor_cfg: SceneEntityCfg specifying the contact sensor.
                    body_names should be ordered as: [left_hand, right_hand, left_foot, right_foot]
        command_name: The name of the command term (only active when moving).
        threshold: Minimum air time to count as "in air".
    
    Returns:
        Reward tensor based on diagonal pair air time synchronization.
    """
    from isaaclab.sensors import ContactSensor
    
    # Get contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get air time for all specified bodies
    # Expected order: [left_hand, right_hand, left_foot, right_foot]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    
    # Check if each limb is in the air
    in_air = last_air_time > threshold  # [num_envs, 4]
    
    # Diagonal pair 1: left_hand (0) + right_foot (3)
    diag1_in_air = in_air[:, 0] & in_air[:, 3]
    
    # Diagonal pair 2: right_hand (1) + left_foot (2)
    diag2_in_air = in_air[:, 1] & in_air[:, 2]
    
    # Reward when either diagonal pair is in the air together
    diagonal_reward = (diag1_in_air | diag2_in_air).float()
    
    # Only reward when there's a velocity command (moving)
    command_term = env.command_manager.get_term(command_name)
    command_norm = torch.norm(command_term.command, dim=1)
    is_moving = (command_norm > 0.1).float()
    
    reward = diagonal_reward * is_moving
    
    return reward


def any_limb_group_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    threshold: float = 0.02,
) -> torch.Tensor:
    """Reward when at least one limb group is in the air.
    
    Four groups: left_hand, left_leg (knee+ankle), right_hand, right_leg (knee+ankle).
    
    Args:
        env: The training environment.
        sensor_cfg: SceneEntityCfg specifying the contact sensor.
                    body_names should be: [left_hand, left_knee, left_ankle, right_hand, right_knee, right_ankle]
        command_name: The name of the command term.
        threshold: Minimum air time to count as "in air".
    
    Returns:
        Reward tensor.
    """
    from isaaclab.sensors import ContactSensor
    
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    
    in_air = last_air_time > threshold  # [num_envs, 6]
    
    # Group 1: left_hand (0)
    # Group 2: left_leg - knee(1) AND ankle(2)
    # Group 3: right_hand (3)
    # Group 4: right_leg - knee(4) AND ankle(5)
    left_hand_air = in_air[:, 0]
    left_leg_air = in_air[:, 1] & in_air[:, 2]
    right_hand_air = in_air[:, 3]
    right_leg_air = in_air[:, 4] & in_air[:, 5]
    
    any_group_air = left_hand_air | left_leg_air | right_hand_air | right_leg_air
    
    command_term = env.command_manager.get_term(command_name)
    command_norm = torch.norm(command_term.command, dim=1)
    is_moving = (command_norm > 0.1).float()
    
    return any_group_air.float() * is_moving

