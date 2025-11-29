"""
Sim2Sim Motion Tracking/Mimic for MuJoCo

This module implements motion tracking functionality using MuJoCo to verify
motion data from datasets. It loads motion sequences and replays them in
MuJoCo simulation with PD control.

Author: Lu Zuxing
Date: 2025/11/13
"""

import re
import time
from pathlib import Path
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
import rich
import torch

from sim2simlib.mimic.config import Sim2SimMimic_Config
from sim2simlib.model.actuator_motor import DCMotor, PIDMotor
from sim2simlib.model.config import Actions
from sim2simlib.model.sim2sim_base import Sim2SimBaseModel
from sim2simlib.utils.motion_dataloader import Motion_Dataloader
from sim2simlib.utils.motion_dataset import Motion_Dataset
from sim2simlib.utils.math import subtract_frame_transforms, matrix_from_quat


class Sim2SimMimic(Sim2SimBaseModel):
    """
    Motion tracking/mimic implementation for MuJoCo.
    
    Inherits from Sim2SimBaseModel to reuse initialization and motor control.
    This class loads motion data from NPZ files and replays them in MuJoCo
    simulation using PD control to track the reference motion.
    
    Args:
        cfg: Sim2SimMimic configuration
    
    Example:
        >>> from sim2simlib.mimic import Sim2SimMimic, create_default_g1_config
        >>> config = create_default_g1_config(
        ...     xml_path="./g1_scene.xml",
        ...     dataset_dirs=["./datasets/OMOMO_Retargeting_Dataset"]
        ... )
        >>> mimic = Sim2SimMimic(cfg=config)
        >>> mimic.view_run()  # Run with viewer
    """
    _cfg: Sim2SimMimic_Config
    
    
    def __init__(self, cfg: Sim2SimMimic_Config):
        # Initialize parent class (Sim2SimBaseModel)
        # This handles: MuJoCo setup, joint names, motor, default pose
        super().__init__(cfg)
        
        # Inherited methods from Sim2SimBaseModel:
        # - _init_joint_names()
        # - _init_actuator_motor()
        # - _init_default_pos_angles()
        # - _init_load_policy() (not used in mimic)
        # - _init_observation_history() (not used in mimic)
        # - get_obs(), get_base_observations() (not used in mimic)
        
        self._init_mimic_dataset()
        self._init_action_scale()
        
    def _init_mimic_dataset(self):
        
        self.motion_id: int = None
        self.time_steps: int = None
        self.global_time_steps: int = None
        
        self.anchor_body_name: str = self._cfg.mimic_dataset_cfg.anchor_body_name
        self.motion_body_names: list[str] = self._cfg.mimic_dataset_cfg.motion_body_names
        self.dataset_body_names: list[str] = self._cfg.mimic_dataset_cfg.dataset_body_names
        self.policy_body_names: list[str] = self._cfg.policy_body_names
        
        
        self.lab_body_indexes = [self.policy_body_names.index(name) for name in self.motion_body_names]
        self.motion_anchor_body_index: int = self.dataset_body_names.index(self.anchor_body_name)
        self.mj_anchor_body_id = self.mj_model.body(self.anchor_body_name).id


        # Load motion dataset
        rich.print(f"[Sim2SimMimic] Loading motion dataset...")
        self.dataset = Motion_Dataset(
            dataset_dirs=self._cfg.mimic_dataset_cfg.dataset_dirs,
            robot_name=self._cfg.mimic_dataset_cfg.robot_name,
            splits=self._cfg.mimic_dataset_cfg.splits
        )
        
        # Create motion dataloader
        rich.print(f"[Sim2SimMimic] Creating motion dataloader...")
        self.dataloader = Motion_Dataloader(
            dataset=self.dataset,
            body_indexes=self.lab_body_indexes,
            device=self._cfg.mimic_dataset_cfg.device
        )
        
        rich.print(f"[Sim2SimMimic] Initialization complete")
        rich.print(f"[Sim2SimMimic] Total motions: {self.dataloader.num_motions}")
        rich.print(f"[Sim2SimMimic] Total frames: {self.dataloader.time_step_total}")
        
    def _init_action_scale(self):
        """Initialize action scaling from configuration."""
        # dict[str, float], such as {'.*_hip_yaw_joint': 0.5475464652142303}
        if isinstance(self._cfg.action_cfg.scale, float):
            self.action_scale = self._cfg.action_cfg.scale
        elif isinstance(self._cfg.action_cfg.scale, dict):
            action_scale_dict = self._cfg.action_cfg.scale
            self.action_scale = []
            for joint_name in self.policy_joint_names:
                scale = 1.0  # default scale
                for pattern, value in action_scale_dict.items():
                    if re.match(pattern, joint_name):
                        scale = value
                        break
                self.action_scale.append(scale)
        
    @property
    def _robot_anchor_pos_w(self) -> np.ndarray:
        """
        Get robot anchor body position in world frame from MuJoCo.
        
        Returns:
            Position np.ndarray
        """
        return self.mj_data.xpos[self.mj_anchor_body_id].copy()  # shape: (3,)
    
    @property
    def _robot_anchor_quat_w(self) -> np.ndarray:
        """
        Get robot anchor body orientation (quaternion) in world frame from MuJoCo.
        
        Returns:
            Quaternion np.ndarray, format order (w, x, y, z)
        """
        return  self.mj_data.xquat[self.mj_anchor_body_id].copy()  # shape: (4,) in (w,x,y,z)
    
    def _obs_motion_command(self) -> np.ndarray:
        """
        Get motion command observation (e.g., target joint positions and velocities).
        
        Returns:
            Motion command observation as a numpy array (R^2N, N is number of joints)
        """
        # Get reference motion for current timestep
        joint_pos_target = self.dataloader.motion_buffer.joint_pos[self.global_time_steps]
        joint_vel_target = self.dataloader.motion_buffer.joint_vel[self.global_time_steps]
        
        # Concatenate joint positions and velocities
        motion_command = torch.cat([joint_pos_target, joint_vel_target], dim=-1)
        
        return motion_command.cpu().numpy()
    
    def _obs_motion_anchor_pos_b(self) -> np.ndarray:
        """
        Get motion anchor position in robot body frame.
        
        Computes the relative position between robot's anchor body and motion's anchor body,
        expressed in robot's anchor body frame.
        
        Returns:
            Position vector in body frame [x, y, z]
        """
        
        anchor_pos_w = self.dataloader.motion_buffer.body_pos_w[self.global_time_steps, self.motion_anchor_body_index] # + self._env.scene.env_origins
        anchor_quat_w = self.dataloader.motion_buffer.body_quat_w[self.global_time_steps, self.motion_anchor_body_index]
        
        # get from mujoco 
        robot_anchor_pos_w = torch.tensor(self._robot_anchor_pos_w).to(anchor_pos_w.device)
        robot_anchor_quat_w = torch.tensor(self._robot_anchor_quat_w).to(anchor_quat_w.device)
        
        pos, _ = subtract_frame_transforms(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            anchor_pos_w,
            anchor_quat_w,
        )
        
        return pos.view(1, -1).cpu().numpy()
    
    def _obs_motion_anchor_ori_b(self) -> np.ndarray:
        """
        Get motion anchor orientation in robot body frame.
        
        Computes the relative orientation between robot's anchor body and motion's anchor body,
        expressed as rotation matrix (first 2 columns flattened).
        
        Returns:
            Rotation matrix first 2 columns flattened [6] (R[:, :2].flatten())
        """
        
        anchor_pos_w = self.dataloader.motion_buffer.body_pos_w[self.global_time_steps, self.motion_anchor_body_index] # + self._env.scene.env_origins
        anchor_quat_w = self.dataloader.motion_buffer.body_quat_w[self.global_time_steps, self.motion_anchor_body_index]
        
        # get from mujoco 
        robot_anchor_pos_w = torch.tensor(self._robot_anchor_pos_w).to(anchor_pos_w.device)
        robot_anchor_quat_w = torch.tensor(self._robot_anchor_quat_w).to(anchor_quat_w.device)

        _, ori = subtract_frame_transforms(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            anchor_pos_w,
            anchor_quat_w,
        )
        mat = matrix_from_quat(ori)
        return mat[..., :2].reshape(mat.shape[0], -1).cpu().numpy()
        
    def _obs_motion_anchor_pos_ori_b(self) -> np.ndarray:
        """
        Get motion anchor position and orientation in robot body frame.
        
        Combines position and orientation observations.
        
        Returns:
            Concatenated position and orientation vector in body frame [9]
        """
        anchor_pos_w = self.dataloader.motion_buffer.body_pos_w[self.global_time_steps, self.motion_anchor_body_index]
        anchor_quat_w = self.dataloader.motion_buffer.body_quat_w[self.global_time_steps, self.motion_anchor_body_index]
        
        # anchor_pos_w  +=  torch.tensor(self.env_origins).to(anchor_pos_w.device)
        robot_anchor_pos_w = torch.tensor(self._robot_anchor_pos_w).to(anchor_pos_w.device)
        robot_anchor_quat_w = torch.tensor(self._robot_anchor_quat_w).to(anchor_quat_w.device)
        
        pos, ori = subtract_frame_transforms(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            anchor_pos_w,
            anchor_quat_w,
        )
        anchor_pos_b = pos.view(-1).cpu().numpy()
        mat = matrix_from_quat(ori)
        anchor_ori_b = mat[..., :2].reshape(-1).cpu().numpy()
        
        return np.concatenate([anchor_pos_b, anchor_ori_b], axis=-1)
    
    def _get_current_motion_observations(self) -> dict[str, np.ndarray]:
        """Get current observations without history processing."""
        motion_observations = {}
        for term in self._cfg.observation_cfg.motion_observations_terms:
            if hasattr(self, f"_obs_{term}"):
                motion_observations[term] = getattr(self, f"_obs_{term}")()
            else:
                raise ValueError(f"Observation term {term} not implemented.")

        # update time steps
        if self.time_steps < self.dataloader.motion_lengths[self.motion_id] - 1:
            self.global_time_steps += 1
            self.time_steps += 1
            
        return motion_observations
    
    def get_obs(self) -> dict[str, np.ndarray]:
        motion_observations = self._get_current_motion_observations()
        base_observations = self.get_base_observations()
        return {**motion_observations, **base_observations}
    
    def reset(self, motion_id):
        # Sample a new motion
        self.motion_id = motion_id
        self.time_steps = 0
        self.global_time_steps = self.dataloader.motion_offsets[self.motion_id]
        
        # lab order
        init_joint_pos = self.dataloader.motion_buffer.joint_pos[self.global_time_steps].cpu().numpy().flatten()
        init_joint_vel = self.dataloader.motion_buffer.joint_vel[self.global_time_steps].cpu().numpy().flatten()
        init_root_pos = self.dataloader.motion_buffer.body_pos_w[self.global_time_steps, 0].cpu().numpy().flatten()
        init_quat = self.dataloader.motion_buffer.body_quat_w[self.global_time_steps, 0].cpu().numpy().flatten()
        
        self.mj_data.qpos[:3] = init_root_pos # + self.env_origins
        self.mj_data.qpos[3:7] = init_quat
        self.mj_data.qpos[self.qpos_maps] = init_joint_pos
        self.mj_data.qvel[self.qvel_maps] = init_joint_vel
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        # get orgins
        # self.env_origins = self._robot_anchor_pos_w
        # self.env_origins[2] = 0.0


    def process_action(self, policy_action: np.ndarray) -> np.ndarray:
        action = policy_action * self.action_scale
        action = np.clip(action, *self._cfg.action_cfg.action_clip) 
        
        joint_pos_action = np.zeros_like(self.init_angles, dtype=np.float32)
        joint_pos_action[self.act_maps] = action
        joint_pos_action += self.init_angles
        return joint_pos_action

    # Run loop for headless simulation
    def headless_run(self):
        counter = 0
        joint_pos_action = self.init_angles
        
        self.reset(motion_id=0)
        while True:
            step_start = time.time()
            
            if counter % self._cfg.control_decimation == 0:
                action = self.act()
                joint_pos_action = self.process_action(action)
                    
            self.apply_action(joint_pos_action)
            mujoco.mj_step(self.mj_model, self.mj_data)
            
            counter += 1
            time_until_next_step = self.mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
    def view_run(self, motion_id: Optional[int] = None):
        counter = 0
        joint_pos_action = self.init_angles
        
        self.reset(motion_id=motion_id)
        
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            while viewer.is_running():
                step_start = time.time()

                if counter % self._cfg.control_decimation == 0:
                    action = self.act()
                    joint_pos_action = self.process_action(action)
                    
                self.apply_action(joint_pos_action)
                mujoco.mj_step(self.mj_model, self.mj_data)
                viewer.sync()  

                counter += 1
                time_until_next_step = self.mj_model.opt.timestep*self.slowdown_factor - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                    
    def view_fk(self, headless: bool = False):
        cur_motion_id = 0
        counter = 0
        self.reset(motion_id=cur_motion_id)
        
        if headless:
            pass
        else:
            with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
                # Configure tracking camera using viewer.cam
                track_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.anchor_body_name)
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = track_id
                viewer.cam.fixedcamid = -1
                viewer.cam.distance = 4.0
                viewer.cam.azimuth = 120.0
                viewer.cam.elevation = -0.0
                viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
        
                while viewer.is_running():
                    step_start = time.time()
                    
                    joint_pos_target = self.dataloader.motion_buffer.joint_pos[self.global_time_steps].cpu().numpy().flatten()
                    joint_vel_target = self.dataloader.motion_buffer.joint_vel[self.global_time_steps].cpu().numpy().flatten()
                    
                    init_root_pos = self.dataloader.motion_buffer.body_pos_w[self.global_time_steps, 0].cpu().numpy().flatten()
                    init_quat = self.dataloader.motion_buffer.body_quat_w[self.global_time_steps, 0].cpu().numpy().flatten()
                    
                    self.mj_data.qpos[:3] = init_root_pos # + self.env_origins
                    self.mj_data.qpos[3:7] = init_quat
                    self.mj_data.qpos[self.qpos_maps] = joint_pos_target
                    self.mj_data.qvel[self.qvel_maps] = joint_vel_target
                    mujoco.mj_forward(self.mj_model, self.mj_data)
                        
                    counter += 1
                    
                    viewer.sync()  

                    # update time steps
                    if self.time_steps < self.dataloader.motion_lengths[self.motion_id] - 1:
                        self.global_time_steps += 1
                        self.time_steps += 1
                    else:
                        cur_motion_id = (cur_motion_id + 1) % self.dataloader.num_motions
                        self.reset(motion_id=cur_motion_id)
                        rich.print(f"[Sim2SimMimic] Switching to motion id: {cur_motion_id}")
                        
                    time_until_next_step = self.mj_model.opt.timestep*self._cfg.control_decimation - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)