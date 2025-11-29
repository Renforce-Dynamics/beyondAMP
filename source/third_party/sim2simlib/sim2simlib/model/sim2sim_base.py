"""
Sim2Sim runner



Author: Lu Zuxing
Data: 2025/8/22
"""
from abc import ABC, abstractmethod
import time
import re
import rich
from dataclasses import dataclass
from glob import glob
import numpy as np
import torch
import mujoco
import mujoco.viewer

from sim2simlib.utils.utils import get_gravity_orientation
from sim2simlib.model.config import Sim2Sim_Config, Actions


class Sim2Sim(ABC):
    qpos_maps: list[int] = []
    qvel_maps: list[int] = []
    act_maps: list[int] = []
    
    init_qpos: np.ndarray
    init_angles: np.ndarray
    
    base_link_id: int = 0
    base_obs_history: dict[str, list[np.ndarray]] = {}
    cmd: list[float]
    last_action: np.ndarray
    _cfg: Sim2Sim_Config

    def __init__(self, cfg: Sim2Sim_Config):
        self._cfg = cfg
        self.xml_path = cfg.xml_path
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = cfg.simulation_dt
        self.mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        self.mj_model.opt.noslip_iterations = 100
        self.slowdown_factor = self._cfg.slowdown_factor
        self.policy_joint_names = self._cfg.policy_joint_names
        self.last_action = np.zeros(len(cfg.policy_joint_names), dtype=np.float32)
        self.cmd = cfg.cmd if cfg.cmd is not None else [0, 0, 0]
        

        
    def _init_joint_names(self):
        self.mujoco_joint_names = [self.mj_model.jnt(i).name for i in range(self.mj_model.njnt)]
        self.qpos_strat_ids = [self.mj_model.jnt_qposadr[i] for i in range(self.mj_model.njnt)]
        self.qvel_strat_ids = [self.mj_model.jnt_dofadr[i] for i in range(self.mj_model.njnt)]

        self.actuators_joint_names = self.mujoco_joint_names[1:]
        rich.print('[INFO] MuJoCo joint names:', self.mujoco_joint_names)
        rich.print('[INFO] Policy joint names:', self.policy_joint_names)
        
        # mujoco order -> policy order
        for joint_name in self.policy_joint_names:
            if joint_name in self.mujoco_joint_names:
                idx = self.mujoco_joint_names.index(joint_name)
                self.qpos_maps.append(self.qpos_strat_ids[idx])
                self.qvel_maps.append(self.qvel_strat_ids[idx])
                self.act_maps.append(idx-1)
            else:
                raise ValueError(f"Joint name {joint_name} not found in MuJoCo model.")

        rich.print("[INFO] qpos maps:", self.qpos_maps)
        rich.print("[INFO] qvel maps:", self.qvel_maps)
        rich.print("[INFO] Action maps:", self.act_maps) 
        
    def _init_load_policy(self):
        self.policy = torch.jit.load(self._cfg.policy_path)
        
    def _init_actuator_motor(self):
        motor_type = self._cfg.motor_cfg.motor_type
        self._cfg.motor_cfg.joint_names = self.actuators_joint_names
        self.DCMotor = motor_type(self._cfg.motor_cfg)  
        
    def _obs_base_lin_vel(self) -> np.ndarray:
        return self.mj_data.qvel[self.base_link_id : self.base_link_id + 3]
    
    def _obs_base_ang_vel(self) -> np.ndarray:
        return self.mj_data.qvel[self.base_link_id + 3 : self.base_link_id + 6]
    
    def _obs_cmd(self) -> np.ndarray:
        return np.array(self.cmd, dtype=np.float32)
    
    def _obs_gravity_orientation(self) -> np.ndarray:
        return get_gravity_orientation(self.mj_data.qpos[self.base_link_id + 3:self.base_link_id + 7])
    
    def _obs_joint_pos(self) -> np.ndarray:
        return (self.mj_data.qpos - self.init_qpos)[self.qpos_maps]
    
    def _obs_joint_vel(self) -> np.ndarray:
        return self.mj_data.qvel[self.qvel_maps]
    
    def _obs_last_action(self) -> np.ndarray:
        return self.last_action

    @abstractmethod
    def act(self) -> np.ndarray:
        pass

    @abstractmethod
    def process_action(self, policy_action: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def apply_action(self, joint_pos_action: np.ndarray):
        pass
    
    @property
    def maped_qvel(self) -> np.ndarray:
        return self.mj_data.qvel[self.qvel_maps]
    
    @property
    def maped_qpos(self) -> np.ndarray:
        return self.mj_data.qpos[self.qpos_maps]
    
    def headless_run(self):
        counter = 0
        joint_pos_action = self.init_angles
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
            
    
    def view_run(self):
        counter = 0
        joint_pos_action = self.init_angles
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
                    
    def debug(self, *args):
        if self._cfg.debug:
            print(*args)

class Sim2SimBaseModel(Sim2Sim):
    
    def __init__(self, cfg: Sim2Sim_Config):
        super().__init__(cfg)  # Call parent constructor
        
        # Initialize functions
        self._init_joint_names()
        self._init_default_pos_angles()
        self._init_load_policy()
        self._init_actuator_motor()
        self._init_observation_history()
    
    def _init_default_pos_angles(self):
        self.mj_data.qpos[:3] = self._cfg.default_pos
        if isinstance(self._cfg.default_angles, np.ndarray):
            self.mj_data.qpos[7:] = self._cfg.default_angles
        elif isinstance(self._cfg.default_angles, dict):
            for joint_name, angle in self._cfg.default_angles.items():
                for i, name in enumerate(self.actuators_joint_names):
                    if re.match(joint_name, name):
                        self.mj_data.qpos[i + 7] = angle
        self.init_qpos = self.mj_data.qpos.copy()
        self.init_angles = self.mj_data.qpos[7:].copy()
        rich.print("[INFO] Initial qpos: ", self.init_qpos)
        rich.print("[INFO] Initial angles: ", self.init_angles)
        rich.print("[INFO] Initial angles mapped: ", self.maped_qpos)

    def _init_observation_history(self):
        """Initialize observation history buffers."""
        if self._cfg.observation_cfg.using_base_obs_history:
            # Get initial observations to determine sizes
            initial_obs = self._get_current_base_observations()
            
            # Initialize history buffers for each observation term
            for term, obs_value in initial_obs.items():
                self.base_obs_history[term] = [obs_value.copy() for _ in range(self._cfg.observation_cfg.base_obs_his_length)]

    def _get_current_base_observations(self) -> dict[str, np.ndarray]:
        """Get current observations without history processing."""
        base_observations = {}
        for term in self._cfg.observation_cfg.base_observations_terms:
            if hasattr(self, f"_obs_{term}"):
                base_observations[term] = getattr(self, f"_obs_{term}")() * self._cfg.observation_cfg.scale[term]
            else:
                raise ValueError(f"Observation term {term} not implemented.")
        return base_observations
    
    def _update_observation_history(self):
        """Update observation history with current observations."""
        if self._cfg.observation_cfg.using_base_obs_history:
            current_obs = self._get_current_base_observations()
            
            for term, obs_value in current_obs.items():
                # Shift history (remove oldest, add newest)
                self.base_obs_history[term] = self.base_obs_history[term][1:] + [obs_value.copy()]         
            
    def get_base_observations(self) -> dict[str, np.ndarray]:
        """Get base observations with optional history."""
        # Update history before getting observations
        if self._cfg.observation_cfg.using_base_obs_history:
            self._update_observation_history()
            
        base_observations = {}
        
        if self._cfg.observation_cfg.using_base_obs_history:
            # Return historical observations
            for term in self._cfg.observation_cfg.base_observations_terms:
                if term in self.base_obs_history:
                    if self._cfg.observation_cfg.base_obs_flatten:
                        # Flatten history: [t-2, t-1, t] -> concatenated array
                        base_observations[term] = np.concatenate(self.base_obs_history[term], axis=-1)
                    else:
                        # Keep as separate timesteps: shape (history_length, obs_dim)
                        base_observations[term] = np.stack(self.base_obs_history[term], axis=0)
                else:
                    raise ValueError(f"Observation term '{term}' not found in history.")
        else:
            # Return current observations without history
            base_observations = self._get_current_base_observations()
        
        return base_observations

    def get_obs(self) -> dict[str, np.ndarray]:
        base_observations = self.get_base_observations()
        return base_observations

    def act(self) -> np.ndarray:
        obs_dict = self.get_obs()
        obs_np = np.concatenate(list(obs_dict.values()), axis=-1).astype(np.float32)
        self.debug("[DEBUG] Observation:", obs_np)
        obs_tensor = torch.from_numpy(obs_np).unsqueeze(0)
        action = self.policy(obs_tensor).detach().numpy().squeeze()
        self.debug("[DEBUG] action:", action)
        self.last_action[:] = action
        return action
    
    def process_action(self, policy_action: np.ndarray) -> np.ndarray:
        action = policy_action * self._cfg.action_cfg.scale
        action = np.clip(action, *self._cfg.action_cfg.action_clip) 
        
        joint_pos_action = np.zeros_like(self.init_angles, dtype=np.float32)
        joint_pos_action[self.act_maps] = action
        joint_pos_action += self.init_angles
        return joint_pos_action

    def apply_action(self, joint_pos_action: np.ndarray):
        tau = self.DCMotor.compute(
                joint_pos=self.mj_data.qpos[self.base_link_id + 7:],
                joint_vel=self.mj_data.qvel[self.base_link_id + 6:],
                action=Actions(
                    joint_pos=joint_pos_action,
                    joint_vel=np.zeros_like(joint_pos_action),
                    joint_efforts=np.zeros_like(joint_pos_action)
                )
            )
        self.mj_data.ctrl[:] = tau

