from __future__ import annotations

import torch
from typing import List
from isaaclab.utils import configclass
from .motion_dataset import MotionDataset, MotionDatasetCfg

class WeightedMotionDataset(MotionDataset):
    """
    Extend MotionDataset with weighted sampling on transition pairs (t, t+1).
    """

    def __init__(
        self,
        cfg: MotionDatasetCfg,
        env,
        device="cpu",
        traj_weights: List[float] | None = None,
        transition_weights: torch.Tensor | None = None,
    ):
        super().__init__(cfg, env, device)

        num_transitions = len(self.index_t)

        if transition_weights is not None:
            assert transition_weights.shape[0] == num_transitions
            self.weights = transition_weights.to(device).clone()
        else:
            self.weights = self._build_transition_weights_from_traj(traj_weights).to(device).clone()

        self._traj_weights = traj_weights
        self.norm_weights()

    # ---------------------------------------------------------
    # Normalization
    # ---------------------------------------------------------
    def norm_weights(self):
        self.weights = self.weights / (self.weights.sum() + 1e-9)

    # ---------------------------------------------------------
    # Build from traj-level weights
    # ---------------------------------------------------------
    def _build_transition_weights_from_traj(self, traj_weights):
        if traj_weights is None:
            return torch.ones(len(self.index_t))

        traj_weights = torch.tensor(traj_weights, dtype=torch.float32)

        weights = []
        for w, L in zip(traj_weights, self._traj_lengths):
            if L >= 2:
                weights.append(torch.full((L - 1,), float(w)))

        return torch.cat(weights, dim=0)

    # ---------------------------------------------------------
    # Dynamic Weight Update API
    # ---------------------------------------------------------

    def update_transition_weights(self, new_weights: torch.Tensor, inplace=True):
        """
        Update weights using full transition-level vector.
        new_weights: shape = (#transitions,)
        """
        assert new_weights.shape[0] == len(self.index_t)
        if inplace:
            self.weights.copy_(new_weights.to(self.device))
        else:
            self.weights = new_weights.to(self.device).clone()

        self.norm_weights()

    def update_traj_weights(self, traj_scores: List[float], inplace=True):
        """
        Update weights using trajectory-level scores (AMP returns, trajectory return, etc.)
        traj_scores: size = num_traj
        """
        assert len(traj_scores) == len(self._traj_lengths)

        tw = self._build_transition_weights_from_traj(traj_scores).to(self.device)
        if inplace:
            self.weights.copy_(tw)
        else:
            self.weights = tw.clone()

        self.norm_weights()

    def update_weights_from_rewards(self, rewards: torch.Tensor, mode="sum", inplace=True):
        """
        Update using rewards.
        rewards: shape = (#transitions,) or (#frames,) for each timestep.
            - If (#frames,), convert to transition-level by ignoring last frame of each trajectory.

        mode: "sum", "mean", or "exp" weighting
        """
        # Case 1: already transition-level rewards
        if rewards.shape[0] == len(self.index_t):
            rw = rewards.to(self.device)
        else:
            # Case 2: frame-level reward → convert to transitions
            assert rewards.shape[0] == self.total_dataset_size
            rewards = rewards.to(self.device)

            rw_list = []
            offset = 0
            for L in self._traj_lengths:
                if L >= 2:
                    rw_list.append(rewards[offset:offset + L - 1])
                offset += L
            rw = torch.cat(rw_list, dim=0)

        if mode == "exp":
            rw = torch.exp(rw)
        elif mode == "mean":
            # mean reward trajectory → map to transitions automatically
            # basically same as sum for transition-level
            pass
        elif mode == "sum":
            pass
        else:
            raise ValueError(f"Unknown reward mode: {mode}")

        return self.update_transition_weights(rw, inplace=inplace)

    # ---------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------
    def sample_batch(self, batch_size: int):
        idx = torch.multinomial(self.weights, batch_size, replacement=True)
        t   = self.index_t[idx]
        tp1 = self.index_tp1[idx]
        return t, tp1


@configclass
class WeightedMotionDatasetCfg(MotionDatasetCfg):
    class_type: type[WeightedMotionDataset] = WeightedMotionDataset
