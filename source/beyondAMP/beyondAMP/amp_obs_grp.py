from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

import beyondAMP.mdp as mdp


@configclass
class AMPObsBaiscCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    
@configclass
class AMPObsSoftTrackingCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    body_quat_w = ObsTerm(func=mdp.body_quat_w)
    body_lin_vel_w = ObsTerm(func=mdp.base_lin_vel)
    body_ang_vel_w = ObsTerm(func=mdp.base_ang_vel)

@configclass
class AMPObsHardTrackingCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    body_pos_w = ObsTerm(func=mdp.body_pose_w)
    body_quat_w = ObsTerm(func=mdp.body_quat_w)
    body_lin_vel_w = ObsTerm(func=mdp.base_lin_vel)
    body_ang_vel_w = ObsTerm(func=mdp.base_ang_vel)
    