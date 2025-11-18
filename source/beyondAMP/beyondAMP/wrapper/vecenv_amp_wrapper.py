# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import torch

from rsl_rl.env import VecEnv

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


class RslRlVecEnvAmpWrapper(VecEnv):
    """Wraps around Isaac Lab environment for RSL-RL library with AMP support

    This wrapper extends the standard RSL-RL wrapper to support Adversarial Motion Priors (AMP).
    It provides methods for accessing AMP observations and handles terminal states for AMP training.

    .. caution::
        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, clip_actions: float | None = None):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.
            clip_actions: The clipping value for actions. If ``None``, then no clipping is done.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )

        # initialize the wrapper
        self.env = env
        self.clip_actions = clip_actions

        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # obtain dimensions of the environment
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        if hasattr(self.unwrapped, "observation_manager"):
            self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
        else:
            self.num_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["policy"])

        # -- privileged observations
        if (
            hasattr(self.unwrapped, "observation_manager")
            and "critic" in self.unwrapped.observation_manager.group_obs_dim
        ):
            self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
        elif hasattr(self.unwrapped, "num_states") and "critic" in self.unwrapped.single_observation_space:
            self.num_privileged_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["critic"])
        else:
            self.num_privileged_obs = 0

        # -- AMP observations
        # Check if environment provides AMP observations
        self._check_amp_support()

        # modify the action space to the clip range
        self._modify_action_space()

        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

        # Initialize buffers for VecEnv interface
        self._init_buffers()

    def _check_amp_support(self):
        """Check if the environment supports AMP observations."""
        # Check if environment has amp_observation_buffer or provides amp_obs in extras
        if hasattr(self.unwrapped, "amp_observation_buffer"):
            # Get AMP observation size from buffer
            if hasattr(self.unwrapped, "amp_observation_size"):
                self.amp_observation_size = self.unwrapped.amp_observation_size
            else:
                # Infer from buffer shape
                buffer_shape = self.unwrapped.amp_observation_buffer.shape
                if len(buffer_shape) == 2:
                    # Shape: (num_envs, amp_observation_size)
                    self.amp_observation_size = buffer_shape[1]
                elif len(buffer_shape) == 3:
                    # Shape: (num_envs, num_amp_observations, amp_observation_space)
                    self.amp_observation_size = buffer_shape[1] * buffer_shape[2]
                else:
                    raise ValueError(
                        f"Unexpected AMP observation buffer shape: {buffer_shape}. "
                        "Expected 2D (num_envs, size) or 3D (num_envs, num_obs, obs_size)."
                    )
        elif hasattr(self.unwrapped, "amp_observation_size"):
            # Environment defines size but no buffer yet
            self.amp_observation_size = self.unwrapped.amp_observation_size
        else:
            # Will try to infer from extras after first observation
            self.amp_observation_size = None

    def _init_buffers(self):
        """Initialize buffers required by VecEnv interface."""
        # Get initial observations to determine buffer sizes
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()

        # Initialize observation buffer
        self.obs_buf = obs_dict["policy"].clone()

        # Initialize privileged observation buffer
        if "critic" in obs_dict:
            self.privileged_obs_buf = obs_dict["critic"].clone()
        elif self.num_privileged_obs > 0:
            self.privileged_obs_buf = torch.zeros(
                (self.num_envs, self.num_privileged_obs), dtype=torch.float32, device=self.device
            )
        else:
            self.privileged_obs_buf = None

        # Initialize reward buffer
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Initialize reset buffer
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Episode length buffer is already in unwrapped env
        self.episode_length_buf = self.unwrapped.episode_length_buf

        # Try to infer AMP observation size from extras if not already set
        if self.amp_observation_size is None:
            if hasattr(self.unwrapped, "extras") and "amp_obs" in self.unwrapped.extras:
                amp_obs = self.unwrapped.extras["amp_obs"]
                if isinstance(amp_obs, torch.Tensor):
                    self.amp_observation_size = amp_obs.shape[1] if len(amp_obs.shape) >= 2 else amp_obs.shape[0]

        # Extras dictionary
        self.extras = {}

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    def get_observations(self) -> torch.Tensor:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        self.obs_buf = obs_dict["policy"]
        return self.obs_buf

    def get_privileged_observations(self) -> torch.Tensor | None:
        """Returns the current privileged observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()

        if "critic" in obs_dict:
            self.privileged_obs_buf = obs_dict["critic"]
            return self.privileged_obs_buf
        else:
            return None

    def get_amp_observations(self) -> torch.Tensor:
        """Returns the current AMP observations of the environment.

        This method extracts AMP observations from the environment's extras dictionary
        or from the environment's AMP observation buffer.

        Returns:
            AMP observations tensor of shape (num_envs, amp_observation_size).

        Raises:
            RuntimeError: If AMP observations are not available in the environment.
        """
        # Try to get from extras first (updated in _get_observations)
        if hasattr(self.unwrapped, "extras") and "amp_obs" in self.unwrapped.extras:
            return self.unwrapped.extras["amp_obs"]
        # Try to get from buffer directly
        elif hasattr(self.unwrapped, "amp_observation_buffer"):
            buffer = self.unwrapped.amp_observation_buffer
            # Handle different buffer shapes
            if len(buffer.shape) == 2:
                # Already flattened: (num_envs, amp_observation_size)
                return buffer
            elif len(buffer.shape) == 3:
                # Shape: (num_envs, num_amp_observations, amp_observation_space)
                return buffer.view(self.num_envs, -1)
            else:
                raise RuntimeError(f"Unexpected AMP observation buffer shape: {buffer.shape}")
        else:
            raise RuntimeError(
                "Environment does not provide AMP observations. "
                "Ensure the environment has 'amp_obs' in extras or 'amp_observation_buffer' attribute."
            )

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self, env_ids: list | torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """Reset the environment.

        Args:
            env_ids: Environment IDs to reset. If None, all environments are reset.

        Returns:
            Tuple of (observations, extras) where observations is the policy observations
            and extras contains additional information including AMP observations.
        """
        # Convert env_ids to tensor if needed
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # Reset the environment
        obs_dict, extras = self.env.reset(env_ids)

        # Update buffers
        self.obs_buf = obs_dict["policy"]
        if "critic" in obs_dict:
            self.privileged_obs_buf = obs_dict["critic"]

        # Store extras
        self.extras = extras.copy()
        self.extras["observations"] = obs_dict

        # Return observations and extras
        return self.obs_buf, self.extras

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, dict, torch.Tensor, torch.Tensor]:
        """Step the environment.

        Args:
            actions: Actions to apply to the environment.

        Returns:
            Tuple of (observations, privileged_observations, rewards, dones, extras, reset_env_ids, terminal_amp_states).
            This extended return signature is required by AMP runners.
        """
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)

        # Update buffers
        self.obs_buf = obs_dict["policy"]
        if "critic" in obs_dict:
            self.privileged_obs_buf = obs_dict["critic"]
        self.rew_buf = rew
        self.reset_buf = dones

        # Get reset environment IDs (environments that need to be reset)
        reset_env_ids = dones.nonzero(as_tuple=False).flatten()

        # Get terminal AMP states for reset environments
        # This needs to be done BEFORE the environment resets, so we get the AMP obs
        # from the current state before reset
        if len(reset_env_ids) > 0:
            try:
                # Get current AMP observations before reset
                # Note: The environment's _get_observations() should have already updated
                # the AMP observation buffer, so we can get it from there
                amp_obs = self.get_amp_observations()
                terminal_amp_states = amp_obs[reset_env_ids].clone()
            except RuntimeError:
                # If AMP observations are not available, create a zero tensor
                # This should not happen if the environment is properly configured
                if self.amp_observation_size is not None:
                    terminal_amp_states = torch.zeros(
                        (len(reset_env_ids), self.amp_observation_size),
                        dtype=torch.float32,
                        device=self.device
                    )
                else:
                    raise RuntimeError(
                        "Cannot create terminal_amp_states: AMP observation size is unknown. "
                        "Ensure the environment provides AMP observations."
                    )
        else:
            # No environments to reset, return empty tensor with correct shape
            if self.amp_observation_size is not None:
                terminal_amp_states = torch.zeros(
                    (0, self.amp_observation_size),
                    dtype=torch.float32,
                    device=self.device
                )
            else:
                # If we don't know the size, try to get it from a sample observation
                try:
                    amp_obs = self.get_amp_observations()
                    terminal_amp_states = torch.zeros(
                        (0, amp_obs.shape[1]),
                        dtype=torch.float32,
                        device=self.device
                    )
                except RuntimeError:
                    raise RuntimeError(
                        "Cannot create terminal_amp_states: AMP observation size is unknown. "
                        "Ensure the environment provides AMP observations."
                    )

        # Store extras
        self.extras = extras.copy()
        self.extras["observations"] = obs_dict
        self.extras["reset_env_ids"] = reset_env_ids
        self.extras["terminal_amp_states"] = terminal_amp_states

        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            self.extras["time_outs"] = truncated

        # Get privileged observations
        privileged_obs = self.privileged_obs_buf if self.num_privileged_obs > 0 else None

        # return the step information with extended signature for AMP
        return self.obs_buf, privileged_obs, self.rew_buf, self.reset_buf, self.extras, reset_env_ids, terminal_amp_states

    def close(self):  # noqa: D102
        return self.env.close()

    """
    Helper functions
    """

    def _modify_action_space(self):
        """Modifies the action space to the clip range."""
        if self.clip_actions is None:
            return

        # modify the action space to the clip range
        # note: this is only possible for the box action space. we need to change it in the future for other
        #   action spaces.
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )

