"""
Motor actuator models for robotic simulation.

This module provides PID and DC motor models with different torque limitation strategies.
The models implement position and velocity control with configurable parameters.

Author: Lu Zuxing
Data: 2025/8/22
"""
import rich
import re
import numpy as np

from sim2simlib.model.config import Actions, Motor_Config


class PIDMotor:
    """
    Simple PID motor model with constant torque limits.
    
    This class implements a basic PID controller for joint position and velocity control.
    The motor applies simple symmetric torque limits regardless of joint velocity.
    
    Mathematical Model:
        τ = K_p * (q_des - q) + K_d * (q̇_des - q̇) + τ_ff
        
        where:
        - τ: Applied torque
        - K_p: Position gain (stiffness)
        - K_d: Velocity gain (damping) 
        - q_des, q̇_des: Desired position and velocity
        - q, q̇: Current position and velocity
        - τ_ff: Feedforward torque
        
    Torque Limits:
        τ_applied = clip(τ, -τ_max, τ_max)
        
        where τ_max is the constant effort limit.
    
    Attributes:
        _effort_limit (np.ndarray): Maximum torque limits for each joint [N⋅m]
        _stiffness (np.ndarray): Position gains (K_p) for each joint [N⋅m/rad]
        _damping (np.ndarray): Velocity gains (K_d) for each joint [N⋅m⋅s/rad]
    """
    _effort_limit: np.ndarray
    _stiffness: np.ndarray
    _damping: np.ndarray

    def __init__(self, cfg: Motor_Config):
        """
        Initialize the PID motor with configuration parameters.
        
        Args:
            cfg (Motor_Config): Motor configuration containing joint names and parameters
        """
        self.cfg = cfg
        self.parse_cfg()
        
    def parse_cfg(self):
        """
        Parse motor configuration and convert parameters to numpy arrays.
        
        This method processes the configuration parameters (effort_limit, stiffness, damping)
        and converts them to per-joint numpy arrays. Supports both scalar values (applied to
        all joints) and dictionaries with regex patterns for joint-specific values.
        
        Prints the parsed parameters for verification.
        """        
        # Parse only the parameters needed for PID control
        self._effort_limit = self._parse_parameter(self.cfg.effort_limit)
        self._stiffness = self._parse_parameter(self.cfg.stiffness)
        self._damping = self._parse_parameter(self.cfg.damping)

        rich.print("[INFO] Motor parameters for ", self.cfg.joint_names)
        rich.print("[INFO] Effort limit: ", self._effort_limit)
        rich.print("[INFO] Stiffness: ", self._stiffness)
        rich.print("[INFO] Damping: ", self._damping)

    def _parse_parameter(self, param: float | dict[str, float]) -> np.ndarray:
        """
        Parse a parameter that can be either a scalar or dictionary with regex patterns.
        
        Args:
            param (float | dict[str, float]): Parameter value(s). Can be:
                - float/int: Single value applied to all joints
                - dict: Regex patterns as keys, values as joint-specific parameters
                
        Returns:
            np.ndarray: Per-joint parameter values with shape (num_joints,)
            
        Raises:
            ValueError: If parameter type is not float, int, or dict
            
        Examples:
            # Scalar value
            param = 100.0  # Applied to all joints
            
            # Dictionary with regex patterns
            param = {
                ".*_hip_.*": 200.0,    # Hip joints get 200.0
                ".*_knee": 150.0,      # Knee joints get 150.0
                ".*": 100.0            # Default for other joints
            }
        """
        num_joints = len(self.cfg.joint_names)
        result = np.zeros(num_joints)
        
        if isinstance(param, (float, int)):
            # If it's a single value, apply to all joints
            result.fill(param)
        elif isinstance(param, dict):
            # If it's a dict with regex patterns, match each joint name
            for i, joint_name in enumerate(self.cfg.joint_names):
                matched = False
                for pattern, value in param.items():
                    if re.match(pattern, joint_name):
                        result[i] = value
                        matched = True
                        break
                if not matched:
                    # If no pattern matches, use a default value (1e-7)
                    result[i] = 1e-7
        else:
            raise ValueError(f"Parameter must be float or dict, got {type(param)}")
            
        return result

    def compute(self, joint_pos: np.ndarray, joint_vel: np.ndarray, action: Actions) -> np.ndarray:
        """
        Compute motor torques using PID control with feedforward.
        
        Implements the PID control law:
            τ = K_p * (q_des - q) + K_d * (q̇_des - q̇) + τ_ff
            
        Args:
            joint_pos (np.ndarray): Current joint positions [rad]
            joint_vel (np.ndarray): Current joint velocities [rad/s]
            action (Actions): Desired actions containing:
                - joint_pos: Desired joint positions [rad]
                - joint_vel: Desired joint velocities [rad/s] 
                - joint_efforts: Feedforward torques [N⋅m]
                
        Returns:
            np.ndarray: Applied motor torques after clipping [N⋅m]
        """
        # Calculate position and velocity errors
        error_pos = action.joint_pos - joint_pos
        error_vel = action.joint_vel - joint_vel
        
        # Compute desired effort using PD control with feedforward
        computed_effort = self._stiffness * error_pos + self._damping * error_vel + action.joint_efforts
        
        # Apply simple torque limits
        applied_effort = self._clip_effort(computed_effort)
        return applied_effort
     
    def _clip_effort(self, effort: np.ndarray) -> np.ndarray:
        """
        Clip motor torques based on constant effort limits.
        
        Applies symmetric torque limits: τ_applied = clip(τ, -τ_max, τ_max)
        
        Args:
            effort (np.ndarray): Computed torques before clipping [N⋅m]
            
        Returns:
            np.ndarray: Clipped torques within effort limits [N⋅m]
        """
        # Ensure effort_limit is numpy array for element-wise operations
        effort_limit = np.asarray(self._effort_limit)
        
        # Clip the torques based on the motor limits
        return np.clip(effort, a_min=-effort_limit, a_max=effort_limit)


class DCMotor(PIDMotor):
    """
    DC motor model with velocity-dependent torque limits.
    
    This class extends the PID motor model to include realistic DC motor characteristics
    with velocity-dependent torque limitations. The model accounts for back-EMF effects
    that reduce available torque at higher speeds.
    
    Mathematical Model:
        τ = K_p * (q_des - q) + K_d * (q̇_des - q̇) + τ_ff
        
    Velocity-Dependent Torque Limits:
        The available torque decreases linearly with velocity due to back-EMF:
        
        v_ratio = q̇ / q̇_max
        
        τ_max(q̇) = τ_sat * (1 - |v_ratio|)    for q̇ ≥ 0
        τ_min(q̇) = -τ_sat * (1 + |v_ratio|)   for q̇ < 0
        
        Final limits:
        τ_max_final = clip(τ_max(q̇), 0, τ_effort_limit)
        τ_min_final = clip(τ_min(q̇), -τ_effort_limit, 0)
        
        τ_applied = clip(τ, τ_min_final, τ_max_final)
        
        where:
        - τ_sat: Saturation torque (no-load torque)
        - q̇_max: Maximum velocity limit
        - τ_effort_limit: Absolute maximum torque limit
        - v_ratio: Normalized velocity ratio
    
    Attributes:
        _saturation_effort (np.ndarray): No-load torque limits [N⋅m]
        _velocity_limit (np.ndarray): Maximum velocities [rad/s]
        _friction (np.ndarray): Friction coefficients [N⋅m⋅s/rad]
        
    Inherits from PIDMotor:
        _effort_limit (np.ndarray): Absolute maximum torque limits [N⋅m]
        _stiffness (np.ndarray): Position gains [N⋅m/rad] 
        _damping (np.ndarray): Velocity gains [N⋅m⋅s/rad]
    """
    _saturation_effort: float | np.ndarray
    _velocity_limit: float | np.ndarray
    _friction: float | np.ndarray

    def __init__(self, cfg: Motor_Config):
        """
        Initialize the DC motor with configuration parameters.
        
        Calls the parent PIDMotor constructor to initialize basic parameters,
        then parses DC motor specific parameters.
        
        Args:
            cfg (Motor_Config): Motor configuration containing joint names and all parameters
        """
        super().__init__(cfg)  # Call parent class constructor
        
    def parse_cfg(self):
        """
        Parse DC motor configuration including velocity-dependent parameters.
        
        First calls the parent parse_cfg() to handle basic PID parameters 
        (effort_limit, stiffness, damping), then parses DC motor specific
        parameters (saturation_effort, velocity_limit, friction).
        
        Prints both inherited and additional parameters for verification.
        """        
        # Call parent class parse_cfg to handle basic parameters
        super().parse_cfg()
        
        # Parse additional DC motor specific parameters
        self._saturation_effort = self._parse_parameter(self.cfg.saturation_effort)
        self._velocity_limit = self._parse_parameter(self.cfg.velocity_limit)
        self._friction = self._parse_parameter(self.cfg.friction)

        rich.print("[INFO] DC Motor additional parameters:")
        rich.print("[INFO] Saturation effort: ", self._saturation_effort)
        rich.print("[INFO] Velocity limit: ", self._velocity_limit)
        rich.print("[INFO] Friction: ", self._friction)

    def compute(self, joint_pos: np.ndarray, joint_vel: np.ndarray, action: Actions) -> np.ndarray:
        """
        Compute motor torques with DC motor characteristics.
        
        Uses the same PID control law as the parent class but applies velocity-dependent
        torque clipping that models realistic DC motor behavior with back-EMF effects.
        
        Args:
            joint_pos (np.ndarray): Current joint positions [rad]
            joint_vel (np.ndarray): Current joint velocities [rad/s]
            action (Actions): Desired actions containing:
                - joint_pos: Desired joint positions [rad]
                - joint_vel: Desired joint velocities [rad/s]
                - joint_efforts: Feedforward torques [N⋅m]
                
        Returns:
            np.ndarray: Applied motor torques after velocity-dependent clipping [N⋅m]
        """
        self._joint_vel = joint_vel
        
        # Calculate position and velocity errors
        error_pos = action.joint_pos - joint_pos
        error_vel = action.joint_vel - joint_vel
        
        # Compute desired effort using PD control with feedforward
        computed_effort = self._stiffness * error_pos + self._damping * error_vel + action.joint_efforts
        
        # Apply motor limits
        applied_effort = self._clip_effort(computed_effort)
        return applied_effort
    
     
    def _clip_effort(self, effort: np.ndarray) -> np.ndarray:
        """
        Clip motor torques based on velocity-dependent DC motor limits.
        
        Implements realistic DC motor torque-speed characteristics where available
        torque decreases linearly with velocity due to back-EMF effects.
        
        Mathematical Implementation:
            v_ratio = q̇ / q̇_max
            
            # Maximum torque in positive direction
            τ_max = τ_sat * (1.0 - v_ratio)
            τ_max = clip(τ_max, 0, τ_effort_limit)
            
            # Maximum torque in negative direction  
            τ_min = τ_sat * (-1.0 - v_ratio)
            τ_min = clip(τ_min, -τ_effort_limit, 0)
            
            # Final clipping
            τ_applied = clip(τ, τ_min, τ_max)
            
        Args:
            effort (np.ndarray): Computed torques before clipping [N⋅m]
            
        Returns:
            np.ndarray: Clipped torques accounting for velocity-dependent limits [N⋅m]
            
        Note:
            Division by zero is handled by setting velocity ratio to 0 when
            velocity limit is 0, effectively disabling velocity dependence.
        """
        # Ensure all parameters are numpy arrays for element-wise operations
        joint_vel = np.asarray(self._joint_vel)
        effort_limit = np.asarray(self._effort_limit)
        saturation_effort = np.asarray(self._saturation_effort)
        velocity_limit = np.asarray(self._velocity_limit)
        
        # Avoid division by zero
        velocity_ratio = np.divide(joint_vel, velocity_limit, 
                                 out=np.zeros_like(joint_vel), 
                                 where=velocity_limit!=0)
        
        # Compute torque limits based on motor characteristics
        # -- max limit (positive direction)
        max_effort = saturation_effort * (1.0 - velocity_ratio)
        max_effort = np.clip(max_effort, a_min=0, a_max=effort_limit)
        
        # -- min limit (negative direction)  
        min_effort = saturation_effort * (-1.0 - velocity_ratio)
        min_effort = np.clip(min_effort, a_min=-effort_limit, a_max=0)

        # Clip the torques based on the computed motor limits
        return np.clip(effort, a_min=min_effort, a_max=max_effort)


