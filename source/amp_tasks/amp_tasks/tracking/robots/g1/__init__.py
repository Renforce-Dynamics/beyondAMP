import gymnasium as gym

from . import rsl_rl_ppo_cfg, flat_env_cfg, amp_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.G1FlatPPORunnerCfg,
    },
)

gym.register(
    id="Tracking-Flat-G1-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.G1FlatWoStateEstimationPPORunnerCfg,
    },
)


gym.register(
    id="Tracking-Flat-G1-AMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": amp_env_cfg.G1AMPTrackFlatEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.G1FlatAMPHardTrackCfg,
    },
)
