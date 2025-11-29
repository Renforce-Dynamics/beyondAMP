import os

from robotlib import ROBOTLIB_ASSETLIB_DIR

MUJOCO_ASSETS = {
    "unitree_go2": os.path.join(ROBOTLIB_ASSETLIB_DIR, "unitree", "unitree_go2", "mjcf", "scene_29dof_rev_1_0.xml"),
    "unitree_g1_29dof": os.path.join(ROBOTLIB_ASSETLIB_DIR, "unitree", "unitree_g1", "mjcf", "scene_29dof_rev_1_0.xml"),
    "unitree_g1_23dof": os.path.join(ROBOTLIB_ASSETLIB_DIR, "unitree", "unitree_g1", "mjcf", "scene_23dof_rev_1_0.xml")
}
    
CHECKPOINT_DIR = "./logs"