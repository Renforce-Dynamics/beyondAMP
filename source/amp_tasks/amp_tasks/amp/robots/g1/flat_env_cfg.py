from isaaclab.utils import configclass

from robotlib.beyondMimic.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from ...amp_env_cfg import AMPEnvCfg

from beyondAMP.obs_groups import AMPObsBaiscCfg, AMPObsBodySoftTrackCfg, AMPObsBodyHardTrackCfg
from beyondAMP.obs_groups import AMPObsBaiscTerms, AMPObsSoftTrackTerms, AMPObsHardTrackTerms
from robotlib.robot_keys.g1_29d import g1_key_body_names

@configclass
class G1FlatEnvCfg(AMPEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        
@configclass
class G1FlatEnvBasicCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.amp = AMPObsBaiscCfg()

@configclass
class G1FlatEnvSoftTrackCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.amp = \
            AMPObsBodySoftTrackCfg().adjust_key_body_indexes(
                # ["body_lin_vel_w"],
                ["body_quat_w", "body_lin_vel_w", "body_ang_vel_w"],
                g1_key_body_names
                )
        
@configclass
class G1FlatEnvHardTrackCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.amp = AMPObsBodyHardTrackCfg().adjust_key_body_indexes(
                [ "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"],
                g1_key_body_names
                )