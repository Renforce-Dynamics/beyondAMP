"""
AMPPPOWithWeightedDataset
---------------------------------------------------------------
An extension of AMPPPO that integrates discriminator-driven
importance weighting on expert transitions. Designed for AMP
(Adversarial Motion Priors) settings where sampling probability
over expert motion transitions can be dynamically adapted.

This implementation:
  • Computes transition-level scores from the discriminator.
  • Converts the scores into positive weights.
  • Updates the WeightedMotionDataset in-place.
  • Keeps the PPO/AMP update loop untouched (clean override).
"""

from __future__ import annotations
import torch
from torch.nn.functional import sigmoid

from beyondAMP.motion.weighted_motion_dataset import WeightedMotionDataset
from .amp_ppo import AMPPPO  # your original AMP+PPO algorithm


class AMPPPOWeighted(AMPPPO):
    """
    AMPPPO extension with expert transition reweighting.
    After every PPO/AMP update, the discriminator signal is used
    to update sampling weights inside the expert dataset.

    Requirements:
      • self.amp_data must be a WeightedMotionDataset
        providing update_weights(new_weights: Tensor).
      • Discriminator takes concatenated (state, next_state).
    """
    amp_data: WeightedMotionDataset
    def __init__(
            self, *args, 
            weight_update_coef: float = 1.0, 
            rescore_interval: int = None,
            **kwargs
        ):
        """
        Args:
            weight_update_coef (float):
                Controls how sharply the discriminator score influences
                sampling weights. The mapping is:
                    w = exp(weight_update_coef * score)
        """
        super().__init__(*args, **kwargs)
        self.weight_update_coef = weight_update_coef
        self.rescore_interval = rescore_interval
        self.rescore_size = None
        # TODO alow not full dataset update
        
        self.rescore_ptr = 0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_transition_scores(self) -> torch.Tensor:
        """
        Evaluate every expert transition using the discriminator.

        Returns:
            Tensor: shape [N_transitions],
                    each entry in [0, 1] (sigmoid(d)).
        """
        if self.rescore_size is None:
            t, tp1 = self.amp_data.index_t, self.amp_data.index_tp1
        else:
            t, tp1 = self.amp_data.sample_batch(self.rescore_size, replacement=False)
        expert_state, expert_next_state = \
            self.amp_data.build_transition(t, tp1)

        # Optional normalization
        if self.amp_normalizer is not None:
            expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
            expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)

        # Concatenate inputs for discriminator
        inp = torch.cat([expert_state, expert_next_state], dim=-1)  # (B, D_s+D_s)
        d = self.discriminator(inp)  # (B,1)

        # Stable scoring function in [0,1]
        score = sigmoid(d).squeeze(-1)
        return score, t, tp1

    # ------------------------------------------------------------------
    @staticmethod
    def _score_to_weight(score: torch.Tensor, coef: float) -> torch.Tensor:
        """
        Convert discriminator scores into positive sampling weights.

        Mapping (commonly used in AMP/GAIL literature):
            w = exp(coef * score)

        Args:
            score: Tensor in [0,1], shape [N_transitions]
            coef: positive scalar controlling sharpness

        Returns:
            Tensor of positive weights.
        """
        return torch.exp(coef * score)

    # ------------------------------------------------------------------
    def update(self):
        """
        Override the high-level update:

        Step 1: Call the standard PPO + AMP update.
        Step 2: Compute discriminator-based scores.
        Step 3: Convert scores → weights and update dataset.
        """
        # -----------------------------
        # Step 1: PPO + AMP update
        # -----------------------------
        log_info = super().update()  # execute original algorithm

        # -----------------------------
        # Step 2: discriminative scoring
        # -----------------------------
        if self.rescore_interval is not None:
            if self.rescore_ptr % self.rescore_ptr == 0:
                scores = self._compute_transition_scores()
                new_weights = self._score_to_weight(scores, self.weight_update_coef)
                self.amp_data.update_weights(new_weights)
                self.rescore_ptr = 0
            self.rescore_ptr += 1

        return log_info
