"""
Sim2Sim Mimic Configuration System

Simplified configuration specifically designed for motion tracking/mimic tasks.
Focuses on motion playback, dataset loading, and PD control parameters.

Author: Lu Zuxing
Date: 2025/11/13
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from sim2simlib.model.config import Sim2Sim_Config, Observations_Config, Actions_Config


@dataclass
class MimicDataset_Config:
    """Dataset configuration for motion loading.
    
    Specifies which datasets and splits to load for motion tracking.
    """
    dataset_dirs: list[str]
    """List of dataset directory paths containing NPZ motion files"""
    
    motion_body_names: list[str]
    
    robot_name: str
    """Robot name (must match directory name in dataset)"""
    
    splits: list[str]
    """Dataset splits to load (e.g., ["train"], ["test"], ["train", "val"])"""
    
    anchor_body_name: str
    """Name of the body to use as motion anchor"""
    
    device: str
    """Device for motion dataloader: "cuda" or "cpu" """
    
    dataset_body_names: Optional[list[str]]


@dataclass
class MimicObservations_Config(Observations_Config):
    motion_observations_terms: list[str]
    
@dataclass
class Sim2SimMimic_Config(Sim2Sim_Config):
    """Configuration for Sim2Sim Mimic model.
    """
    
    mimic_dataset_cfg: MimicDataset_Config
    observation_cfg: MimicObservations_Config
    policy_body_names: Optional[list[str]]
    