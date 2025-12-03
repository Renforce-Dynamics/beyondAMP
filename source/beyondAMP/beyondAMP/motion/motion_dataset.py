from __future__ import annotations

import os
import numpy as np
import torch
from typing import Sequence, List, Union
# from beyondAMP.obs_groups import AMPObsBaiscCfg

from isaaclab.utils import configclass
from dataclasses import MISSING
from .motion_transition import MotionTransition

class MotionDataset:
    """
    Load multiple motion files and build (s_t, s_{t+1}) transitions.
    Efficient contiguous tensors + pre-built index mapping.
    """

    def __init__(
        self, 
        cfg: MotionDatasetCfg,
        env,
        device: str = "cpu",
        ):
        self.cfg = cfg
        self.env = env
        self.device = device
        self.robot = env.scene[cfg.asset_name]
        self.motion_files = cfg.motion_files
        self.observation_terms = cfg.amp_obs_terms
        
        body_names = cfg.body_names
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(body_names, preserve_order=True)[0], dtype=torch.long, device=device
        )

        anchor_name = cfg.anchor_name
        self.anchor_index = torch.tensor(
            self.robot.find_bodies(anchor_name, preserve_order=True)[0], dtype=torch.long, device=device
        )
        
        self.load_motions()
        self.init_observation_dims()

    def load_motions(self):
        # Storage lists (later concatenated)
        joint_pos_list = []
        joint_vel_list = []
        body_pos_w_list = []
        body_quat_w_list = []
        body_lin_vel_w_list = []
        body_ang_vel_w_list = []
        fps_list = []
        traj_lengths = []

        # Load all motion files
        for f in self.motion_files:
            assert os.path.isfile(f), f"Invalid motion file: {f}"
            data = np.load(f)

            fps_list.append(float(data["fps"]))
            traj_len = data["joint_pos"].shape[0]
            traj_lengths.append(traj_len)

            joint_pos_list.append(torch.tensor(data["joint_pos"], dtype=torch.float32))
            joint_vel_list.append(torch.tensor(data["joint_vel"], dtype=torch.float32))
            body_pos_w_list.append(torch.tensor(data["body_pos_w"], dtype=torch.float32))
            body_quat_w_list.append(torch.tensor(data["body_quat_w"], dtype=torch.float32))
            body_lin_vel_w_list.append(torch.tensor(data["body_lin_vel_w"], dtype=torch.float32))
            body_ang_vel_w_list.append(torch.tensor(data["body_ang_vel_w"], dtype=torch.float32))

        # Concatenate all trajectories into single big tensors
        self.joint_pos      = torch.cat(joint_pos_list, dim=0).to(self.device)
        self.joint_vel      = torch.cat(joint_vel_list, dim=0).to(self.device)
        self.body_pos_w_all      = torch.cat(body_pos_w_list, dim=0).to(self.device)
        self.body_quat_w_all     = torch.cat(body_quat_w_list, dim=0).to(self.device)
        self.body_lin_vel_w_all  = torch.cat(body_lin_vel_w_list, dim=0).to(self.device)
        self.body_ang_vel_w_all  = torch.cat(body_ang_vel_w_list, dim=0).to(self.device)

        self.total_dataset_size = sum(traj_lengths)

        # Keep per-trajectory FPS if needed
        self.fps_list = fps_list

        # Build transition index list: (global_index_t, global_index_t+1)
        self.index_t, self.index_tp1 = self._build_transition_indices(traj_lengths, self.device)

    # ----------------------- Property API -----------------------
    
    def subtract_flaten(self, target: torch.Tensor):
        target = target[:, self.body_indexes]
        return target.reshape(self.total_dataset_size, -1)
    
    @property
    def body_pos_w(self):
        return self.body_pos_w_all[:, self.body_indexes].reshape(self.total_dataset_size, -1)
    @property
    def body_quat_w(self):
        return self.body_quat_w_all[:, self.body_indexes].reshape(self.total_dataset_size, -1)
    @property
    def body_lin_vel_w(self):
        return self.body_lin_vel_w_all[:, self.body_indexes].reshape(self.total_dataset_size, -1)
    @property
    def body_ang_vel_w(self):
        return self.body_ang_vel_w_all[:, self.body_indexes].reshape(self.total_dataset_size, -1)
    
    @property
    def anchor_pos_w(self):
        return self.body_pos_w_all[:, self.anchor_index].reshape(self.total_dataset_size, -1)
    @property
    def anchor_quat_w(self):
        return self.body_quat_w_all[:, self.anchor_index].reshape(self.total_dataset_size, -1)
    @property
    def anchor_lin_vel_w(self):
        return self.body_lin_vel_w_all[:, self.anchor_index].reshape(self.total_dataset_size, -1)
    @property
    def anchor_ang_vel_w(self):
        return self.body_ang_vel_w_all[:, self.anchor_index].reshape(self.total_dataset_size, -1)
        

    # ----------------------- Transition index builder -----------------------

    def observation_dim_cast(self, name)->int:
        # shape_cast_table = {
        #     "displacement": self.body_indexes.shape[-1]
        # }
        if hasattr(self, name):
            obs_term: torch.Tensor = getattr(self, name)
            assert isinstance(obs_term, torch.Tensor), f"invalid observation name: {name} for get dim"
            return obs_term.shape[-1]
        else:
            raise NotImplementedError(f"Failed for term: {name}")

    def init_observation_dims(self):
        observation_dims = []
        for obs_term in self.observation_terms:
            # observation_terms.append(obs_term)
            observation_dims.append(self.observation_dim_cast(obs_term))
        self.observation_dim = sum(observation_dims)
        self.observation_dims = observation_dims

    # ----------------------- Transition index builder -----------------------

    def _build_transition_indices(self, traj_lengths: List[int], device: str):
        """
        Build valid (t, t+1) pairs without crossing trajectory boundaries.
        """
        idx_t = []
        idx_tp1 = []

        offset = 0
        for L in traj_lengths:
            if L < 2:
                offset += L
                continue
            t = torch.arange(offset, offset + L - 1)
            idx_t.append(t)
            idx_tp1.append(t + 1)
            offset += L

        idx_t = torch.cat(idx_t).to(device)
        idx_tp1 = torch.cat(idx_tp1).to(device)
        return idx_t, idx_tp1

    # ----------------------- Batch Sampling API -----------------------

    def sample_batch(self, batch_size: int):
        """
        Sample a batch of transitions:
            s_t → s_{t+1}

        Returns dict:
            {
                "joint_pos_t": ...,
                "joint_pos_tp1": ...,
                ...
            }
        """
        idx = torch.randint(0, len(self.index_t), (batch_size,), device=self.device)
        t = self.index_t[idx]
        tp1 = self.index_tp1[idx]
        return t, tp1

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for idx in range(0, num_mini_batch):
            res_t, res_tp1 = [], []
            t, tp1 = self.sample_batch(mini_batch_size)
            for term in self.observation_terms:
                _t, _tp1 = getattr(self, term)[t], getattr(self, term)[tp1]
                res_t.append(_t); res_tp1.append(_tp1)
            res_t, res_tp1 = torch.cat(res_t, dim=-1), torch.cat(res_tp1, dim=-1)
            yield res_t, res_tp1

@configclass
class MotionDatasetCfg:
    class_type          : type[MotionDataset] = MotionDataset
    asset_name          : str = "robot"
    motion_files        : List[str] = MISSING
    body_names          : List[str] = MISSING
    amp_obs_terms       : List[str] = MISSING
    anchor_name         : str = MISSING
