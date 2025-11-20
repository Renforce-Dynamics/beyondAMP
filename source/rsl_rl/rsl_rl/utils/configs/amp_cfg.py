from typing import List
from isaaclab.utils import configclass
from rsl_rl.utils.configs.rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from beyondAMP.amp_obs_grp import AMPObsBaiscCfg
from rsl_rl.runners.amp_on_policy_runner import AMPOnPolicyRunner

from dataclasses import MISSING

@configclass
class AMPPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):    
    class_name="AMPPPO"
    amp_replay_buffer_size: int = 100000

@configclass
class AMPDataCfg:
    asset_name: str = "robot"
    motion_files: List[str] = MISSING
    body_names: List[str] = MISSING
    amp_obs_terms:  List[str] = ["joint_pos", "joint_vel"]

@configclass
class AMPRunnerCfg(RslRlOnPolicyRunnerCfg):
    runner_type:             type[AMPOnPolicyRunner] = AMPOnPolicyRunner
    amp_data:               AMPDataCfg = MISSING
    amp_reward_coef:        float = MISSING
    amp_discr_hidden_dims:  List[int] = MISSING
    amp_task_reward_lerp:   float = 0.9
    amp_min_normalized_std: float = 0.0 # recmended for no minimal explore std. since the action limit may large 