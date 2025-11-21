from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands.null_command import NullCommand
from isaaclab.envs.mdp.commands.commands_cfg import NullCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class EpisodeSuccessCommand(NullCommand):
    """Command generator that tracks episode success rate.
    
    Inherits from NullCommand but adds episode success rate tracking functionality.
    """

    def __init__(self, cfg: EpisodeSuccessCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the success tracking command generator."""
        super().__init__(cfg, env)
        
        # buffer to record success information
        self.cur_success_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.last_success_buf = torch.zeros_like(self.cur_success_buf)
        
        # metrics to track success rates
        self.metrics["step_success_rate"] = torch.zeros(1, device=self.device)
        self.metrics["episode_success_rate"] = torch.zeros(1, device=self.device)
        self.metrics["alive_time_rate"] = torch.zeros(self.num_envs, device=self.device)

    def compute(self, dt: float):
        self._update_metrics()
        
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        
        # resolve the environment IDs
        if env_ids is None:
            env_ids = slice(None)
            
        # add logging metrics
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            if metric_value.ndim == 0 or (metric_value.ndim > 0 and metric_value.shape[0] == 1):
                extras[metric_name] = metric_value.item()
            else:
                extras[metric_name] = torch.mean(metric_value[env_ids]).item()
        return extras
    
    def _update_metrics(self):
        
        # get the terminations info
        reset_terminated = self._env.termination_manager.terminated # (num_envs, )
        reset_time_outs = self._env.termination_manager.time_outs # (num_envs, )
        reset_dones = self._env.termination_manager.dones # (num_envs, )

        self.metrics["step_success_rate"] = torch.sum(reset_time_outs.float()) \
            / (torch.sum(reset_dones.float()) + 1e-6)
            
        # record the success information
        self.last_success_buf[reset_dones] = self.cur_success_buf[reset_dones]
        self.cur_success_buf[reset_time_outs] = True
        self.cur_success_buf[reset_terminated] = False
        
        self.metrics["episode_success_rate"] = torch.sum(self.last_success_buf.float()) / self._env.num_envs
        
        # get the episode info 
        episode_length_buf = self._env.episode_length_buf # (num_envs, )
        max_episode_length = self._env.max_episode_length
        self.metrics["alive_time_rate"] = episode_length_buf.float() / max_episode_length
        

@configclass
class EpisodeSuccessCommandCfg(NullCommandCfg):
    """Configuration for the episode success tracking command generator."""
    
    class_type: type = EpisodeSuccessCommand
    
