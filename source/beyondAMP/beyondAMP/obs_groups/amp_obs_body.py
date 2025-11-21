from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

import beyondAMP.mdp as mdp

from .amp_obs_base import AMPObsBaseCfg

# Terms with only body reference motion
    
"""
Example
```
@configclass
class G1FlatEnvSoftTrackCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.amp = \
            AMPObsSoftTrackCfg().adjust_key_body_indexes(
                ["body_quat_w", "body_lin_vel_w", "body_ang_vel_w"],
                g1_key_body_names
                )
```
"""
    
@configclass
class AMPObsBodySoftTrackCfg(AMPObsBaseCfg):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    body_quat_w = ObsTerm(func=mdp.body_quat_w)
    body_lin_vel_w = ObsTerm(func=mdp.body_lin_vel_w)
    body_ang_vel_w = ObsTerm(func=mdp.body_ang_vel_w)

AMPObsSoftTrackTerms = ["joint_pos", "joint_vel", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]

@configclass
class AMPObsBodyHardTrackCfg(AMPObsBaseCfg):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    body_pos_w = ObsTerm(func=mdp.body_pos_w)
    body_quat_w = ObsTerm(func=mdp.body_quat_w)
    body_lin_vel_w = ObsTerm(func=mdp.body_lin_vel_w)
    body_ang_vel_w = ObsTerm(func=mdp.body_ang_vel_w)
    
AMPObsHardTrackTerms = ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]
