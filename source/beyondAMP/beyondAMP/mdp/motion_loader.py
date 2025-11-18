"""Generic Motion Loader for AMP training with humanoid robots."""

import os
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Sequence, Union
from pathlib import Path

try:
    from pybullet_utils import transformations
except ImportError:
    transformations = None


class MotionLoader:
    """Motion loader for AMP training with support for multiple trajectories."""

    # Default data dimensions (can be overridden in config)
    DEFAULT_POS_SIZE = 3
    DEFAULT_ROT_SIZE = 4
    DEFAULT_JOINT_POS_SIZE = 21  # Typical humanoid
    DEFAULT_LINEAR_VEL_SIZE = 3
    DEFAULT_ANGULAR_VEL_SIZE = 3
    DEFAULT_JOINT_VEL_SIZE = 21

    def __init__(
        self,
        device: str,
        time_between_frames: float,
        data_dir: str = '',
        preload_transitions: bool = False,
        num_preload_transitions: int = 1000000,
        motion_files: Optional[List[str]] = None,
        body_indexes: Optional[Sequence[int]] = None,  # Body indices to use (like beyondMimic)
        config: Optional[Dict] = None,
    ):
        """Initialize the motion loader.

        Args:
            device: Device to store tensors ('cpu' or 'cuda')
            time_between_frames: Time between consecutive frames in seconds
            data_dir: Directory containing motion files
            preload_transitions: Whether to preload transition samples
            num_preload_transitions: Number of transitions to preload
            motion_files: List of motion file paths (if None, auto-discover .npz files)
            body_indexes: Body indices to use (like beyondMimic), if None use root body (0)
            config: Configuration dictionary for data dimensions and structure
        """
        self.device = device
        self.time_between_frames = time_between_frames
        self.body_indexes = body_indexes or [0]  # Default to root body

        # Load configuration
        self.config = config or self._get_default_config()
        self._parse_config()

        # Initialize data structures for AMP training
        self.trajectories = []  # AMP observations (joint_pos, joint_vel, root_height)
        self.trajectories_full = []  # Full state data
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Trajectory length in seconds
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        # Auto-discover motion files if not provided
        if motion_files is None:
            motion_files = self._discover_motion_files(data_dir)

        # Load all motion data
        for motion_file in motion_files:
            self._load_motion_file(motion_file)

        # Setup trajectory weights for sampling
        if self.trajectory_weights:
            self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
            self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
            self.trajectory_lens = np.array(self.trajectory_lens)
            self.trajectory_num_frames = np.array(self.trajectory_num_frames)

            # Preload transitions if requested
            if preload_transitions:
                self._preload_transitions(num_preload_transitions)

            # Stack all trajectories for batch processing
            if self.trajectories_full:
                self.all_trajectories_full = torch.vstack(self.trajectories_full)
        else:
            logging.warning("No motion files loaded successfully")

    # =====================================================================
    # Configuration
    # =====================================================================
# -----------------------------------------------------------------------------
# Configurations
# -----------------------------------------------------------------------------
    # robot_type = "r2b"
    # robot_cfg, motion_align_cfg = load_robot_cfg(robot_type)
    # motion_name = ("/home/idlaber24/code/Isaaclab_workspace/labBundle/tasks/trackerLab/data/configs/GMR/lafan_r2_wholebody/test.yaml")

    def _get_default_config(self) -> Dict:
        """Get default configuration for humanoid robot.
        
        Note: Sizes will be inferred from actual data when first motion file is loaded.
        """
        return {
            'pos_size': None,  # Will be inferred from data
            'rot_size': None,  # Will be inferred from data
            'joint_pos_size': None,  # Will be inferred from data
            'linear_vel_size': None,  # Will be inferred from data
            'angular_vel_size': None,  # Will be inferred from data
            'joint_vel_size': None,  # Will be inferred from data
            # Data structure indices
            'root_pos_start': 0,
            'root_rot_start': None,  # Will be set based on pos_size
            'joint_pos_start': None,  # Will be set based on rot_size
            'linear_vel_start': None,
            'angular_vel_start': None,
            'joint_vel_start': None,
        }

    def _parse_config(self):
        """Parse and validate configuration."""
        # Only calculate indices if sizes are already set
        # Otherwise, they will be calculated after first data load
        if all(self.config.get(key) is not None for key in 
               ['pos_size', 'rot_size', 'joint_pos_size', 'linear_vel_size', 
                'angular_vel_size', 'joint_vel_size']):
            self._update_config_indices()
        
        # Set observation dimension (AMP observations exclude root pos/rot)
        # observation_dim = joint_pos_size + joint_vel_size + 1 (root height)
        # This will be recalculated after loading trajectories to match actual data
        self._observation_dim = None
    
    def _update_config_indices(self):
        """Update configuration indices based on current size values."""
        self.config['root_rot_start'] = self.config['pos_size']
        self.config['joint_pos_start'] = self.config['root_rot_start'] + self.config['rot_size']
        self.config['linear_vel_start'] = self.config['joint_pos_start'] + self.config['joint_pos_size']
        self.config['angular_vel_start'] = self.config['linear_vel_start'] + self.config['linear_vel_size']
        self.config['joint_vel_start'] = self.config['angular_vel_start'] + self.config['angular_vel_size']
    
    def _infer_config_from_data(self, root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel):
        """Infer configuration sizes from actual data shapes.
        
        Args:
            root_pos: Root position array [num_frames, 3]
            root_quat: Root quaternion array [num_frames, 4]
            joint_pos: Joint positions array [num_frames, num_joints]
            root_lin_vel: Root linear velocity array [num_frames, 3]
            root_ang_vel: Root angular velocity array [num_frames, 3]
            joint_vel: Joint velocities array [num_frames, num_joints]
        """
        # Only update if not already set (first file loaded)
        if self.config['pos_size'] is None:
            self.config['pos_size'] = root_pos.shape[1] if len(root_pos.shape) > 1 else root_pos.shape[0]
        if self.config['rot_size'] is None:
            self.config['rot_size'] = root_quat.shape[1] if len(root_quat.shape) > 1 else root_quat.shape[0]
        if self.config['joint_pos_size'] is None:
            self.config['joint_pos_size'] = joint_pos.shape[1] if len(joint_pos.shape) > 1 else joint_pos.shape[0]
        if self.config['linear_vel_size'] is None:
            self.config['linear_vel_size'] = root_lin_vel.shape[1] if len(root_lin_vel.shape) > 1 else root_lin_vel.shape[0]
        if self.config['angular_vel_size'] is None:
            self.config['angular_vel_size'] = root_ang_vel.shape[1] if len(root_ang_vel.shape) > 1 else root_ang_vel.shape[0]
        if self.config['joint_vel_size'] is None:
            self.config['joint_vel_size'] = joint_vel.shape[1] if len(joint_vel.shape) > 1 else joint_vel.shape[0]
        
        # Update indices after inferring sizes
        self._update_config_indices()
        
        logging.info(f"Inferred config from data: pos_size={self.config['pos_size']}, "
                    f"rot_size={self.config['rot_size']}, "
                    f"joint_pos_size={self.config['joint_pos_size']}, "
                    f"linear_vel_size={self.config['linear_vel_size']}, "
                    f"angular_vel_size={self.config['angular_vel_size']}, "
                    f"joint_vel_size={self.config['joint_vel_size']}")

    # =====================================================================
    # File loading
    # =====================================================================

    def _discover_motion_files(self, data_dir: str) -> List[str]:
        """Auto-discover NPZ motion files in directory."""
        if not data_dir:
            return []

        data_path = Path(data_dir)
        if not data_path.exists():
            logging.warning(f"Data directory {data_dir} does not exist")
            return []

        # Only search for NPZ files
        motion_files = list(data_path.glob('**/*.npz'))
        return [str(f) for f in motion_files]

    def _load_motion_file(self, motion_file: str):
        """Load motion data from a single NPZ file."""
        try:
            if not motion_file.endswith('.npz'):
                logging.warning(f"Unsupported file format: {motion_file}. Only .npz files are supported.")
                return

            motion_data, metadata = self._load_npz_file(motion_file)

            # Process motion data
            self._process_motion_data(motion_data, metadata, motion_file)

        except Exception as e:
            logging.error(f"Failed to load motion file {motion_file}: {e}")

    def _load_npz_file(self, motion_file: str) -> tuple:
        """Load motion data from NPZ file (following beyondMimic pattern)."""
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)

        # Extract data (following beyondMimic pattern)
        self.fps = data.get("fps", 30.0)
        joint_pos = data["joint_pos"]  # [num_frames, num_joints]
        joint_vel = data["joint_vel"]  # [num_frames, num_joints]

        # FK-processed body data
        body_pos_w = data["body_pos_w"]    # [num_frames, num_bodies, 3]
        body_quat_w = data["body_quat_w"]  # [num_frames, num_bodies, 4]
        body_lin_vel_w = data["body_lin_vel_w"]  # [num_frames, num_bodies, 3]
        body_ang_vel_w = data["body_ang_vel_w"]  # [num_frames, num_bodies, 3]

        num_frames = joint_pos.shape[0]

        # Select bodies using body_indexes (like beyondMimic)
        # Default to root body (index 0) for AMP training
        root_body_idx = self.body_indexes[0] if self.body_indexes else 0

        # Extract root body data for state representation
        root_pos = body_pos_w[:, root_body_idx, :]      # [num_frames, 3]
        root_quat = body_quat_w[:, root_body_idx, :]    # [num_frames, 4]
        root_lin_vel = body_lin_vel_w[:, root_body_idx, :]  # [num_frames, 3]
        root_ang_vel = body_ang_vel_w[:, root_body_idx, :]  # [num_frames, 3]

        # Normalize and standardize quaternions
        for f_i in range(num_frames):
            if transformations is not None:
                root_quat[f_i] = self._standardize_quaternion(
                    self._quaternion_normalize(root_quat[f_i])
                )

        # Infer configuration from actual data (only on first file)
        self._infer_config_from_data(root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel)

        # Stack data in the expected format: [root_pos, root_rot, joint_pos, root_lin_vel, root_ang_vel, joint_vel]
        motion_data = np.column_stack([
            root_pos,      # Root position (3)
            root_quat,     # Root rotation quaternion (4, wxyz)
            joint_pos,     # Joint positions
            root_lin_vel,  # Root linear velocity (3)
            root_ang_vel,  # Root angular velocity (3)
            joint_vel      # Joint velocities
        ])

        # Extract metadata
        metadata = {
            'motion_weight': float(data.get('motion_weight', 1.0)),
            'frame_duration': 1.0 / self.fps,
            'num_frames': num_frames
        }

        return motion_data, metadata

    def _process_motion_data(self, motion_data: np.ndarray, metadata: Dict, motion_file: str):
        """Process loaded motion data."""
        # Ensure config is set (should be set by _infer_config_from_data in _load_npz_file)
        if not all(self.config.get(key) is not None for key in 
                   ['pos_size', 'rot_size', 'joint_pos_size', 'linear_vel_size', 
                    'angular_vel_size', 'joint_vel_size']):
            raise RuntimeError("Configuration not properly initialized. "
                             "This should not happen if data is loaded correctly.")
        
        # Validate data dimensions
        expected_dim = self._get_expected_data_dimension()
        if motion_data.shape[1] != expected_dim:
            logging.warning(f"Motion data dimension mismatch in {motion_file}: "
                          f"expected {expected_dim}, got {motion_data.shape[1]}")
            return

        # Store trajectory metadata
        self.trajectory_names.append(Path(motion_file).stem)
        self.trajectory_idxs.append(len(self.trajectory_idxs))

        # Extract AMP observations (exclude root pos/rot, only joint_pos + joint_vel)
        # This matches rsl_rl AMPLoader: from JOINT_POSE_START to JOINT_VEL_END
        joint_start = self.config['joint_pos_start']
        joint_end = self.config['joint_vel_start'] + self.config['joint_vel_size']
        amp_data = motion_data[:, joint_start:joint_end]
        
        # Convert to tensors
        self.trajectories.append(torch.tensor(amp_data, dtype=torch.float32, device=self.device))
        self.trajectories_full.append(torch.tensor(motion_data, dtype=torch.float32, device=self.device))

        # Store metadata
        self.trajectory_weights.append(metadata['motion_weight'])
        self.trajectory_frame_durations.append(metadata['frame_duration'])
        traj_len = (motion_data.shape[0] - 1) * metadata['frame_duration']
        self.trajectory_lens.append(traj_len)
        self.trajectory_num_frames.append(metadata['num_frames'])

        logging.info(f"Loaded {traj_len:.2f}s motion from {motion_file}")

    def _get_expected_data_dimension(self) -> int:
        """Get expected dimension of motion data."""
        return self.config['joint_vel_start'] + self.config['joint_vel_size']

    # =====================================================================
    # Quaternion utilities
    # =====================================================================

    def _standardize_quaternion(self, q: np.ndarray) -> np.ndarray:
        """Standardize quaternion by ensuring w >= 0."""
        return -q if q[-1] < 0 else q

    def _quaternion_normalize(self, q: np.ndarray) -> np.ndarray:
        """Normalize quaternion to unit length."""
        norm = np.linalg.norm(q)
        if np.isclose(norm, 0.0):
            # Return identity quaternion [0, 0, 0, 1] for wxyz format
            identity = np.zeros_like(q)
            identity[-1] = 1.0
            return identity
        return q / norm

    # =====================================================================
    # Preloading
    # =====================================================================

    def _preload_transitions(self, num_transitions: int):
        """Preload transition samples for faster training."""
        logging.info(f'Preloading {num_transitions} transitions')
        traj_idxs = self.weighted_traj_idx_sample_batch(num_transitions)
        times = self.traj_time_sample_batch(traj_idxs)
        self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
        self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
        logging.info('Finished preloading')

    # =====================================================================
    # Sampling methods
    # =====================================================================
    def weighted_traj_idx_sample(self) -> int:
        """Sample trajectory index based on weights."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size: int) -> np.ndarray:
        """Batch sample trajectory indices."""
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx: int) -> float:
        """Sample random time for trajectory."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(0, self.trajectory_lens[traj_idx] * np.random.uniform() - subst)

    def traj_time_sample_batch(self, traj_idxs: np.ndarray) -> np.ndarray:
        """Sample random times for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    # =====================================================================
    # Interpolation methods
    # =====================================================================

    def slerp(self, val0: torch.Tensor, val1: torch.Tensor, blend: float) -> torch.Tensor:
        """Spherical linear interpolation."""
        return (1.0 - blend) * val0 + blend * val1

    def quaternion_slerp(self, q1: torch.Tensor, q2: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation for quaternions."""
        if transformations is None:
            return self.slerp(q1, q2, blend)

        # Convert to numpy for quaternion operations
        q1_np = q1.cpu().numpy()
        q2_np = q2.cpu().numpy()
        blend_np = blend.cpu().numpy()

        result = []
        for i in range(len(q1_np)):
            try:
                interpolated = transformations.quaternion_slerp(q1_np[i], q2_np[i], blend_np[i, 0])
                result.append(self._standardize_quaternion(interpolated))
            except:
                result.append(self.slerp(q1_np[i], q2_np[i], blend_np[i, 0]))

        return torch.tensor(np.array(result), device=self.device, dtype=torch.float32)

    # =====================================================================
    # Frame retrieval methods
    # =====================================================================
    def get_frame_at_time(self, traj_idx: int, time: float) -> torch.Tensor:
        """Get interpolated frame at specific time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_full_frame_at_time(self, traj_idx: int, time: float) -> torch.Tensor:
        """Get full interpolated frame at specific time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs: np.ndarray, times: np.ndarray) -> torch.Tensor:
        """Batch get interpolated frames."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)

        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)

        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]

        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time_batch(self, traj_idxs: np.ndarray, times: np.ndarray) -> torch.Tensor:
        """Batch get full interpolated frames."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)

        # Initialize tensors for batch processing
        batch_size = len(traj_idxs)
        all_frame_pos_starts = torch.zeros(batch_size, self.config['pos_size'], device=self.device)
        all_frame_pos_ends = torch.zeros(batch_size, self.config['pos_size'], device=self.device)
        all_frame_rot_starts = torch.zeros(batch_size, self.config['rot_size'], device=self.device)
        all_frame_rot_ends = torch.zeros(batch_size, self.config['rot_size'], device=self.device)
        
        joint_start = self.config['joint_pos_start']
        joint_end = self.config['joint_vel_start'] + self.config['joint_vel_size']
        amp_size = joint_end - joint_start
        all_frame_amp_starts = torch.zeros(batch_size, amp_size, device=self.device)
        all_frame_amp_ends = torch.zeros(batch_size, amp_size, device=self.device)

        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = self.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = self.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = self.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = self.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][:, joint_start:joint_end]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][:, joint_start:joint_end]

        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = self.quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)

        return torch.cat([pos_blend, rot_blend, amp_blend], dim=-1)

    def get_frame(self) -> torch.Tensor:
        """Get random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self) -> torch.Tensor:
        """Get random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames: int) -> torch.Tensor:
        """Get batch of random full frames."""
        if hasattr(self, 'preloaded_s'):
            idxs = np.random.choice(len(self.preloaded_s), size=num_frames, replace=True)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0: torch.Tensor, frame1: torch.Tensor, blend: float) -> torch.Tensor:
        """Linearly interpolate between two frames."""
        root_pos0, root_pos1 = self.get_root_pos(frame0), self.get_root_pos(frame1)
        root_rot0, root_rot1 = self.get_root_rot(frame0), self.get_root_rot(frame1)
        joints0, joints1 = self.get_joint_pose(frame0), self.get_joint_pose(frame1)
        linear_vel_0, linear_vel_1 = self.get_linear_vel(frame0), self.get_linear_vel(frame1)
        angular_vel_0, angular_vel_1 = self.get_angular_vel(frame0), self.get_angular_vel(frame1)
        joint_vel_0, joint_vel_1 = self.get_joint_vel(frame0), self.get_joint_vel(frame1)

        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)

        # Quaternion interpolation
        if transformations is not None:
            blend_root_rot = transformations.quaternion_slerp(
                root_rot0.cpu().numpy(), root_rot1.cpu().numpy(), blend)
            blend_root_rot = torch.tensor(
                self._standardize_quaternion(blend_root_rot),
                dtype=torch.float32, device=self.device)
        else:
            blend_root_rot = self.slerp(root_rot0, root_rot1, blend)

        blend_joints = self.slerp(joints0, joints1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([
            blend_root_pos, blend_root_rot, blend_joints,
            blend_linear_vel, blend_angular_vel, blend_joints_vel])

    # =====================================================================
    # AMP training generator
    # =====================================================================

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        """Generate batches of AMP transitions."""
        for _ in range(num_mini_batch):
            if hasattr(self, 'preloaded_s'):
                idxs = np.random.choice(len(self.preloaded_s), size=mini_batch_size, replace=True)
                s = self._extract_amp_observation(self.preloaded_s[idxs])
                s_next = self._extract_amp_observation(self.preloaded_s_next[idxs])
            else:
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                
                # Get AMP frames (joint_pos + joint_vel, without root height)
                s = self.get_frame_at_time_batch(traj_idxs, times)
                s_next = self.get_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
                
                # Get root height (z coordinate) from full frames
                full_s = self.get_full_frame_at_time_batch(traj_idxs, times)
                full_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
                
                # Add root height to AMP observations
                root_height_idx = self.config['root_pos_start'] + 2
                s = torch.cat([s, full_s[:, root_height_idx:root_height_idx + 1]], dim=-1)
                s_next = torch.cat([s_next, full_s_next[:, root_height_idx:root_height_idx + 1]], dim=-1)

            yield s, s_next

    def _extract_amp_observation(self, full_frame: torch.Tensor) -> torch.Tensor:
        """Extract AMP observation (joint_pos + joint_vel + root_height) from full frame."""
        joint_start = self.config['joint_pos_start']
        joint_end = self.config['joint_vel_start'] + self.config['joint_vel_size']
        root_height_idx = self.config['root_pos_start'] + 2
        
        joint_data = full_frame[:, joint_start:joint_end]
        root_height = full_frame[:, root_height_idx:root_height_idx + 1]
        return torch.cat([joint_data, root_height], dim=-1)

    # =====================================================================
    # Data access methods
    # =====================================================================

    def get_root_pos(self, pose: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Get root position from pose."""
        return pose[self.config['root_pos_start']:self.config['root_pos_start'] + self.config['pos_size']]

    def get_root_pos_batch(self, poses: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Get root positions from batch of poses."""
        start = self.config['root_pos_start']
        return poses[:, start:start + self.config['pos_size']]

    def get_root_rot(self, pose: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Get root rotation from pose."""
        return pose[self.config['root_rot_start']:self.config['root_rot_start'] + self.config['rot_size']]

    def get_root_rot_batch(self, poses: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Get root rotations from batch of poses."""
        start = self.config['root_rot_start']
        return poses[:, start:start + self.config['rot_size']]

    def get_joint_pose(self, pose: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Get joint positions from pose."""
        return pose[self.config['joint_pos_start']:self.config['joint_pos_start'] + self.config['joint_pos_size']]

    def get_joint_pose_batch(self, poses: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Get joint positions from batch of poses."""
        start = self.config['joint_pos_start']
        return poses[:, start:start + self.config['joint_pos_size']]

    def get_linear_vel(self, pose: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Get root linear velocity from pose."""
        return pose[self.config['linear_vel_start']:self.config['linear_vel_start'] + self.config['linear_vel_size']]

    def get_linear_vel_batch(self, poses: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Get root linear velocities from batch of poses."""
        start = self.config['linear_vel_start']
        return poses[:, start:start + self.config['linear_vel_size']]

    def get_angular_vel(self, pose: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Get root angular velocity from pose."""
        return pose[self.config['angular_vel_start']:self.config['angular_vel_start'] + self.config['angular_vel_size']]

    def get_angular_vel_batch(self, poses: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Get root angular velocities from batch of poses."""
        start = self.config['angular_vel_start']
        return poses[:, start:start + self.config['angular_vel_size']]

    def get_joint_vel(self, pose: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Get joint velocities from pose."""
        return pose[self.config['joint_vel_start']:self.config['joint_vel_start'] + self.config['joint_vel_size']]

    def get_joint_vel_batch(self, poses: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Get joint velocities from batch of poses."""
        start = self.config['joint_vel_start']
        return poses[:, start:start + self.config['joint_vel_size']]

    # =====================================================================
    # Properties
    # =====================================================================

    @property
    def observation_dim(self) -> int:
        """Size of AMP observations (joint_pos + joint_vel + root_height)."""
        if self._observation_dim is None:
            if self.trajectories:
                # observation_dim = trajectories[0].shape[1] + 1 (root height)
                self._observation_dim = self.trajectories[0].shape[1] + 1
            else:
                # Fallback to calculated dimension if no trajectories loaded
                self._observation_dim = (self.config['joint_pos_size'] + 
                                       self.config['joint_vel_size'] + 1)
        return self._observation_dim

    @property
    def num_motions(self) -> int:
        """Number of loaded motion trajectories."""
        return len(self.trajectory_names)