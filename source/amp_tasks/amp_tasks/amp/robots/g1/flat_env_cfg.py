from isaaclab.utils import configclass

from robotlib.beyondMimic.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from ...amp_env_cfg import AMPEnvCfg


@configclass
class G1FlatEnvCfg(AMPEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
