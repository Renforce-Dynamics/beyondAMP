from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands.null_command import NullCommand
from isaaclab.envs.mdp.commands.commands_cfg import NullCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ActionFluctuationRatioCommand(NullCommand):
    """Command generator that tracks action fluctuation ratio.
    
    Inherits from NullCommand but adds action fluctuation ratio tracking functionality.
    
    AFR = 1/T ∑‖aₜ - aₜ₋₁‖ = 1/T * sum(||a_t - a_{t-1}||)
    """

    def __init__(self, cfg: ActionFluctuationRatioCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the action fluctuation ratio command generator."""
        super().__init__(cfg, env)
        
        # buffer to record action fluctuations, (num_envs, max_episode_length, action_dim)
        # self.max_episode_length = self._env.max_episode_length 
        # self.action_dim = self._env.action_manager.action_term_dim
        # self.action_buf = torch.zeros(self.num_envs, self.max_episode_length, self.action_dim, device=self.device)
        self.step_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # metrics to track action fluctuation ratio
        self.metrics["action_fluctuation_ratio"] = torch.zeros(self.num_envs, device=self.device)

    def compute(self, dt: float):
        self._update_metrics()
        self.step_counter += 1
        
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        
        # resolve the environment IDs
        if env_ids is None:
            env_ids = slice(None)
            
        # add logging metrics
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            # compute the mean metric value
            extras[metric_name] = torch.mean(metric_value[env_ids]).item()
            # reset the metric value
            metric_value[env_ids] = 0.0
        
        # reset the buffer
        self.step_counter[env_ids] = 0

        return extras
    
    def _update_metrics(self):
        # get valid environments (not the first step)
        valid_envs = self.step_counter > 0
        
        if torch.any(valid_envs):
            # get the current and previous actions info
            if self.cfg.using_term:
                cur_action = self._env.action_manager._terms[self.cfg.term_name]._raw_actions
                prev_action = self._env.action_manager._terms[self.cfg.term_name]._prev_raw_actions
            else:
                cur_action = self._env.action_manager._action # (num_envs, action_dim)
                prev_action = self._env.action_manager._prev_action # (num_envs, action_dim)
            
            # compute action differences and their norms
            action_diff = cur_action - prev_action  # (num_envs, action_dim)
            action_diff_abs = torch.abs(action_diff)  # (num_envs, action_dim)
            action_diff_norm = torch.norm(action_diff_abs, dim=-1)  # (num_envs,)
            
            # update cumulative AFR for valid environments
            # AFR = 1/T * sum(||a_t - a_{t-1}||)
            # We compute running average: new_avg = (old_avg * (t-1) + new_value) / t
            current_step = self.step_counter.float()
            self.metrics["action_fluctuation_ratio"][valid_envs] = (
                self.metrics["action_fluctuation_ratio"][valid_envs] * (current_step[valid_envs] - 1) + 
                action_diff_norm[valid_envs]
            ) / current_step[valid_envs]
        
@configclass
class ActionFluctuationRatioCommandCfg(NullCommandCfg):
    """Configuration for the episode success tracking command generator."""
    
    class_type: type = ActionFluctuationRatioCommand
    
    using_term: bool = False
    term_name: str = "joint_pos"