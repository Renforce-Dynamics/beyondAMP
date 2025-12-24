from isaaclab.utils import configclass

from robotlib.beyondMimic.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from ...tracking_env_cfg import TrackingEnvCfg
from amp_tasks.amp_task_demo_data_cfg import soccer_shoot_file, demo_punch_file

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
        self.commands.motion.motion_file = demo_punch_file

@configclass
class G1FlatNoRelaEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.motion_anchor_ori_b = None # using gravity instead
        self.observations.policy.base_lin_vel = None

@configclass
class G1FlatWoStateEstimationEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.projected_gravity = None
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
        