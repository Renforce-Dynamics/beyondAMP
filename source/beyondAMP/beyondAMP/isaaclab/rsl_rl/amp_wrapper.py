import torch
from .vecenv_wrapper import RslRlVecEnvWrapper
from beyondAMP.motion.motion_dataset import MotionDataset
from beyondAMP.motion.weighted_motion_dataset import WeightedMotionDataset

class AMPEnvWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, clip_actions = None, *, motion_dataset=None):
        super().__init__(env, clip_actions)
        self.rewards_shape = self.unwrapped.reward_manager._step_reward.shape[-1]
        if isinstance(motion_dataset, MotionDataset) or motion_dataset is None:
            self.motion_dataset = motion_dataset
        else:
            self.motion_dataset = WeightedMotionDataset(motion_dataset, env.unwrapped, env.unwrapped.device)
            
        self.unwrapped.motion_dataset = self.motion_dataset
    
    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return obs_dict["policy"]
    
    def get_amp_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return obs_dict["amp"]
    
    def step(self, actions, *, not_amp=True, **kwargs):
        if not_amp:
            return super().step(actions, **kwargs)
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        obs = obs_dict["policy"]
        privileged_obs = obs_dict.get("critic", obs)
        terminal_amp_states = obs_dict.get("amp", obs)
        extras["observations"] = obs_dict
        reset_env_ids = torch.where(dones)[0]
        # extras["terminated"] = terminated
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # return the step information
        return obs, privileged_obs, rew, dones, extras, reset_env_ids, terminal_amp_states[reset_env_ids]
    
    @property
    def dof_pos_limits(self)->torch.Tensor:
        return self.unwrapped.scene["robot"].data.joint_pos_limits