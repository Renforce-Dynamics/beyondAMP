import os
import numpy as np
import torch
from typing import Sequence, List, Union
from beyondAMP.amp_obs import AMPObsBaiscCfg

class MotionDataset:
    """
    Load multiple motion files and build (s_t, s_{t+1}) transitions.
    Efficient contiguous tensors + pre-built index mapping.
    """

    def __init__(
        self, 
        motion_files: Sequence[str], body_indexes: Sequence[str], 
        amp_obs_cfg: Union[AMPObsBaiscCfg|dict],
        device: str = "cpu",
        ):
        self.device = device
        self.body_indexes = torch.tensor(body_indexes, dtype=torch.long)
        self.amp_obs_cfg = amp_obs_cfg

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
        for f in motion_files:
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
        self.joint_pos = torch.cat(joint_pos_list, dim=0).to(device)
        self.joint_vel = torch.cat(joint_vel_list, dim=0).to(device)
        self.body_pos_w_all = torch.cat(body_pos_w_list, dim=0).to(device)
        self.body_quat_w_all = torch.cat(body_quat_w_list, dim=0).to(device)
        self.body_lin_vel_w_all = torch.cat(body_lin_vel_w_list, dim=0).to(device)
        self.body_ang_vel_w_all = torch.cat(body_ang_vel_w_list, dim=0).to(device)

        # Keep per-trajectory FPS if needed
        self.fps_list = fps_list

        # Build transition index list: (global_index_t, global_index_t+1)
        self.index_t, self.index_tp1 = self._build_transition_indices(traj_lengths, device)
        self.build_observation()

    # ----------------------- Transition index builder -----------------------

    @classmethod
    def from_cfg(cls, cfg: dict, env, device):
        body_names = cfg["body_names"]
        robot = env.scene[cfg["asset_name"]]
        body_indexes = torch.tensor(
            robot.find_bodies(body_names, preserve_order=True)[0], dtype=torch.long, device=device
        )
        obj = cls(
            motion_files = cfg["motion_files"],
            body_indexes = body_indexes,
            amp_obs_cfg  = cfg["amp_obs_cfg"],
            device       = device
            )
        return obj

    def observation_dim_cast(self, name)->int:
        shape_cast_table = {
            "displacement": self.body_indexes.shape[-1]
        }
        
        if hasattr(self, name):
            obs_term: torch.Tensor = getattr(self, name)
            assert isinstance(obs_term, torch.Tensor), f"invalid observation name: {name} for get dim"
            return obs_term.shape[-1]
        else:
            return shape_cast_table[name]

    def build_observation(self):
        observation_terms = []
        observation_dims = []
        
        obs_dict  = self.amp_obs_cfg if isinstance(self.amp_obs_cfg, dict) else self.amp_obs_cfg()
        for obs_term in obs_dict.keys():
            observation_terms.append(obs_term)
            observation_dims.append(self.observation_dim_cast(obs_term))

        self.observation_dim = sum(observation_dims)
        self.observation_dims = observation_dims
        self.observation_terms = observation_terms

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

    # ----------------------- Body selection -----------------------

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.body_pos_w_all[:, self.body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.body_quat_w_all[:, self.body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_w_all[:, self.body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_w_all[:, self.body_indexes]

    # ----------------------- Batch Sampling API -----------------------

    def sample(self, batch_size: int):
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

        return {
            "joint_pos": (self.joint_pos[t],  self.joint_pos[tp1]),
            "joint_vel": (self.joint_vel[t], self.joint_vel[tp1]),
            "body_pos_w": 
                (self.body_pos_w_all[t][:, self.body_indexes], self.body_pos_w_all[tp1][:, self.body_indexes]),
            "body_quat_w": 
                (self.body_quat_w_all[t][:, self.body_indexes], self.body_quat_w_all[tp1][:, self.body_indexes]),
            "body_lin_vel_w": 
                (self.body_lin_vel_w_all[t][:, self.body_indexes], self.body_lin_vel_w_all[tp1][:, self.body_indexes]),
            "body_ang_vel_w": 
                (self.body_ang_vel_w_all[t][:, self.body_indexes], self.body_ang_vel_w_all[tp1][:, self.body_indexes]),
        }


    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for idx in range(0, num_mini_batch):
            sample = self.sample(mini_batch_size)
            t1, tp1 = [], []
            for term in self.observation_terms:
                _t1, _tp1 = sample[term]
                t1.append(_t1); tp1.append(_tp1)
                
            t1 = torch.cat(t1, dim=-1)
            tp1 = torch.cat(tp1, dim=-1)
            yield t1, tp1