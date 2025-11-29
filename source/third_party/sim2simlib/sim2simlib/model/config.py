"""
sim2simlib config system
follow the isaaclab manager system,
support:
1. Observations Terms order define,
2. Observations Terms history,
3. Motor Joint name Regular expression,

Author: Lu Zuxing
Data: 2025/8/22
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Actions():
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    joint_efforts: np.ndarray

@dataclass
class Actions_Config:
    scale: float | dict[str, float]
    action_clip: tuple[float, float]

@dataclass
class Observations_Config:
    base_observations_terms: list[str]
    scale: dict[str, float]
    using_base_obs_history: bool
    base_obs_flatten: bool
    """
    using base_obs_flatten=True, terms flatten as N*dim(prev->current); 
    else base_obs_flatten=False, terms stack shape as (N, dim)
    """
    base_obs_his_length: int
    

@dataclass
class Motor_Config(): 
    motor_type: type = None
    """Motor type: PIDMotor or DCMotor"""
    joint_names: list[str] = None
    """joint_names are the mujoco order actuator names, must same as mujoco joint defined."""
    
    effort_limit: float | dict[str, float] = 0.0
    """Max output effort"""
    stiffness: float | dict[str, float] = 0.0
    """PID control: P Gain"""
    damping: float | dict[str, float] = 0.0
    """PID control: D Gain"""
    
    saturation_effort: float | dict[str, float] = 0.0
    """DCMotor: Max output effort"""
    velocity_limit: float | dict[str, float] = 0.0
    """DCMotor: Max output velocity"""
    friction: float | dict[str, float] = 0.0
    """not use"""

@dataclass
class Sim2Sim_Config:
    
    robot_name: str
    """robot name for exp"""
    simulation_dt: float
    """physics simulation time step, pid control dt"""
    control_decimation: int
    """policy simulation time step"""
    slowdown_factor: float
    """slowdown factor for debug view run, biger is slower"""
    
    xml_path: str
    """robot xml with scene"""
    policy_path: str
    """export policy as *.pt format"""
    policy_joint_names: list[str]
    """isaaclab policy joint order"""
    
    default_pos: np.ndarray
    """Issaclab scene/robot/init_state/pos"""
    default_angles: np.ndarray | dict[str, float]
    """Issaclab scene/robot/init_state/joint_pos"""
    
    observation_cfg: Observations_Config
    """observation config for obs order and scale"""
    action_cfg: Actions_Config
    """action config for clip and scale"""
    motor_cfg: Motor_Config
    """actuators config"""
    
    debug: bool
    """debug print key information"""

    cmd: list[float]
    """base sim2sim command for velocity task"""
