from isaaclab.utils import configclass

from robotlib.beyondMimic.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from ...tracking_env_cfg import TrackingEnvCfg

from beyondAMP.obs_groups import AMPObsBaiscCfg, AMPObsBodySoftTrackCfg, AMPObsBodyHardTrackCfg
from beyondAMP.obs_groups import AMPObsBaiscTerms, AMPObsSoftTrackTerms, AMPObsHardTrackTerms

from robotlib.robot_keys.g1_29d import g1_key_body_names, g1_anchor_name

@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]
        # self.observations.policy.projected_gravity = None
        self.commands.motion.debug_vis = False
        self.commands.motion.motion_file = "data/datasets/MocapG1Full/LAFAN/walk1_subject1.npz"
        
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.motion_anchor_ori_b = None # using gravity instead

@configclass
class G1AMPTrackFlatEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.amp = AMPObsBodyHardTrackCfg().adjust_key_body_indexes(
                [ "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"],
                g1_key_body_names
                )
        self.observations.policy.command = None