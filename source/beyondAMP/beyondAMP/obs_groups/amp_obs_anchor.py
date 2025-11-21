from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

import beyondAMP.mdp as mdp

from .amp_obs_base import AMPObsBaseCfg

# Terms with anchor reference motion

"""
Example, note that the anchor and the body shares the same func
```
@configclass
class G1FlatEnvSoftTrackCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.amp = \
            AMPObsSoftTrackCfg().adjust_key_body_indexes(
                ["body_quat_w", "body_lin_vel_w", "body_ang_vel_w"],
                g1_anchor_name
                )
```
"""

@configclass
class AMPObsAnchorHardTrackCfg(AMPObsBaseCfg):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    anchor_pos_w = ObsTerm(func=mdp.anchor_pos_w)
    anchor_quat_w = ObsTerm(func=mdp.anchor_quat_w)
    anchor_lin_vel_w = ObsTerm(func=mdp.anchor_lin_vel_w)
    anchor_ang_vel_w = ObsTerm(func=mdp.anchor_ang_vel_w)
    
AMPObsAnchorHardTrackTerms = \
    ["joint_pos", "joint_vel", "anchor_pos_w", "anchor_quat_w", "anchor_lin_vel_w", "anchor_ang_vel_w"]