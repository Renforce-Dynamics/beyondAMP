from __future__ import annotations

import os
import numpy as np
import torch
from typing import Sequence, List, Union, Tuple
from dataclasses import dataclass, MISSING

@dataclass
class MotionTransition:
    joint_pos:      Tuple[torch.Tensor, torch.Tensor]
    joint_vel:      Tuple[torch.Tensor, torch.Tensor]
    body_pos_w:     Tuple[torch.Tensor, torch.Tensor]
    body_quat_w:    Tuple[torch.Tensor, torch.Tensor]
    body_lin_vel_w: Tuple[torch.Tensor, torch.Tensor]
    body_ang_vel_w: Tuple[torch.Tensor, torch.Tensor]