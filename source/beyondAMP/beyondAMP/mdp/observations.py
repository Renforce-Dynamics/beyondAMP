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
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
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