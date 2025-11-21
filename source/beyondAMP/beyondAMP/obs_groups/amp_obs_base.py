from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

import beyondAMP.mdp as mdp

"""
Not that you should update the 
"""
@configclass
class AMPObsBaseCfg(ObsGroup):
    def adjust_key_body_indexes(self, terms:list, key_bodys:list):
        for term_name in terms:
            term:ObsTerm = getattr(self, term_name)
            if "asset_cfg" in term.params:
                term.params["asset_cfg"].body_names = key_bodys
            else:
                term.params["asset_cfg"] = SceneEntityCfg(name="robot", body_names=key_bodys)
        return self

@configclass
class AMPObsBaiscCfg(AMPObsBaseCfg):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    
AMPObsBaiscTerms = ["joint_pos", "joint_vel"]


