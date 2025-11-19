from typing import List
from isaaclab.utils import configclass
from rsl_rl.utils.configs.rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from beyondAMP.amp_obs import AMPObsBaiscCfg

from dataclasses import MISSING

@configclass
class AMPPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    amp_replay_buffer_size: int = 100000

@configclass
class AMPDataCfg:
    motion_files: List[str] = MISSING
    body_indexes: List[str] = MISSING
    amp_obs_cfg:  AMPObsBaiscCfg = MISSING

@configclass
class AMPRunnerCfg(RslRlOnPolicyRunnerCfg):
    amp_data:               AMPDataCfg = MISSING
    amp_reward_coef:        float = MISSING
    amp_discr_hidden_dims:  List[int] = MISSING
    amp_task_reward_lerp:   float = 0.9