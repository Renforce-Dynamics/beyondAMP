from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def body_quat_w(
    env:ManagerBasedRLEnv, 
    asset_cfg:SceneEntityCfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]
    return body_quat
    

def amp_obs_body_displacement(env:ManagerBasedRLEnv, asset_cfg:SceneEntityCfg=SceneEntityCfg("robot")):
    return