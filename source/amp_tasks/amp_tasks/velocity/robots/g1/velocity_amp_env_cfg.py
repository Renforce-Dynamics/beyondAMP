
from isaaclab.utils import configclass
from .velocity_env_cfg import RobotEnvCfg

from beyondAMP.obs_groups import AMPObsBaiscCfg, AMPObsBodySoftTrackCfg, AMPObsBodyHardTrackCfg


@configclass
class G1VelocityAMPEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.amp = AMPObsBaiscCfg()