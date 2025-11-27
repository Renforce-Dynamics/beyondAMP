from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def body_pos_w(
    env:ManagerBasedRLEnv, 
    asset_cfg:SceneEntityCfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids] - env.scene.env_origins.unsqueeze(1)
    return body_pos_w.reshape(env.num_envs, -1)

def body_quat_w(
    env:ManagerBasedRLEnv, 
    asset_cfg:SceneEntityCfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]
    return body_quat.reshape(env.num_envs, -1)
    
def body_lin_vel_w(
    env:ManagerBasedRLEnv, 
    asset_cfg:SceneEntityCfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_lin_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
    return body_lin_vel_w.reshape(env.num_envs, -1)

def body_ang_vel_w(
    env:ManagerBasedRLEnv, 
    asset_cfg:SceneEntityCfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_ang_vel_w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids]
    return body_ang_vel_w.reshape(env.num_envs, -1)

anchor_pos_w = body_pos_w
anchor_quat_w = body_quat_w
anchor_lin_vel_w = body_lin_vel_w
anchor_ang_vel_w  = body_ang_vel_w


def base_lin_vel_yaw(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Base linear velocity in yaw-only frame (ignores pitch/roll).
    
    Suitable for crawling robots where body is tilted but velocity should
    be measured relative to the horizontal facing direction.
    
    Returns:
        Linear velocity [vx, vy, vz] in yaw-only frame.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Get root linear velocity in world frame
    root_lin_vel_w = asset.data.root_lin_vel_w
    # Get yaw-only quaternion (ignores pitch/roll)
    yaw_quat = math_utils.yaw_quat(asset.data.root_quat_w)
    # Transform to yaw-only frame
    root_lin_vel_yaw = math_utils.quat_rotate_inverse(yaw_quat, root_lin_vel_w)
    return root_lin_vel_yaw


def base_ang_vel_yaw(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Base angular velocity in yaw-only frame (ignores pitch/roll).
    
    Suitable for crawling robots where body is tilted but angular velocity
    should be measured relative to the horizontal facing direction.
    
    Returns:
        Angular velocity [wx, wy, wz] in yaw-only frame.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Get root angular velocity in world frame
    root_ang_vel_w = asset.data.root_ang_vel_w
    # Get yaw-only quaternion (ignores pitch/roll)
    yaw_quat = math_utils.yaw_quat(asset.data.root_quat_w)
    # Transform to yaw-only frame
    root_ang_vel_yaw = math_utils.quat_rotate_inverse(yaw_quat, root_ang_vel_w)
    return root_ang_vel_yaw