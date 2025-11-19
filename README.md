# beyondAMP: One-Step Integration of AMP into IsaacLab

## Overview

**beyondAMP** provides a unified pipeline to integrate Adversarial Motion Priors (AMP) into any IsaacLab robot setup, with minimal modifications and full compatibility with custom robot designs. [中文README](./README_cn.md)

## 🚀 Fast Setup

```bash
cd beyondAMP
bash scripts/setup_ext.sh
# Downloads assets, robot configs, and installs dependencies
```

Optional VSCode workspace setup:

```bash
python scripts/setup_vscode.py
```

## 📌 How to Use

### Quick Start

* Basic environment: `source/amp_tasks/amp_tasks/amp`
* PPO config for G1 robot: `source/amp_tasks/amp_tasks/amp/robots/g1/rsl_rl_ppo_cfg.py`

Training can be launched with:
```
scripts/factoryIsaac/train.py --task AMPG1 --headless
```

To evaluate or visualize a trained checkpoint:
```
scripts/factoryIsaac/play.py --headless --target <path to your ckpt.pt> --video --num_envs 32
```

| Description             | Video      |
| ----------------------- | ---------- |
| AMP Training Preview    | *(insert)* |
| Motion Matching Example | *(insert)* |

### Dataset Preparation

The dataset follows the same structure and conventions used in BeyondMimic(whole_body_tracking). All motion sequences should be stored as *.npz files and placed under data/datasets/, maintaining a consistent directory layout with the reference pipeline.

For motion retargeting and preprocessing, GMR is recommended for generating high-quality retargeted mocap data. TrackerLab may be used to perform forward kinematics checks and robot-specific adjustments, ensuring the motions remain physically plausible for your robot model.

With these tools, the dataset organization naturally aligns with the conventions established in BeyondMimic(whole_body_tracking), enabling seamless integration with the AMP training pipeline.

> Following the dataset pipeline of **BeyondMimic**:
> 
> * Motion files: place `*.npz` into `data/datasets/`
> * Recommended tools:
>   * **GMR** for retargeted motion
>   * **TrackerLab** for FK validation & robot-specific preprocessing

### AMP Integration Details

* AMP observation group added via a new `amp` observation config
* RSL-RL integration: `source/rsl_rl/rsl_rl/env/isaaclab/amp_wrapper.py`
* Default transition builder: `source/beyondAMP/beyondAMP/amp_obs.py`

> For full tutorial and customization, see `docs/tutorial.md`.


<details>
<summary><strong>Additional Notes</strong></summary>

* Fully modular AMP observation builder
* Compatible with IsaacLab 4.5+
* Designed for rapid experimentation across robot morphologies

</details>

## 🙏 Acknowledgement

### Referenced Repositories

| Repository                                                           | Purpose                               |
| -------------------------------------------------------------------- | ------------------------------------- |
| [robotlib](https://github.com/Renforce-Dynamics/robotlib)            | Robot configurations                  |
| [assetslib](https://github.com/Renforce-Dynamics/assetslib)          | Asset storage                         |
| [TrackerLab](https://github.com/Renforce-Dynamics/trackerLab)        | Data organization & retargeting tools |
| [AMP_for_hardware](https://github.com/escontra/AMP_for_hardware)     | AMP implementation reference          |
| [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) | Dataset format & tracking comparison  |

---

## 📘 Citation

```bibtex
@software{zheng2025@beyondAMP,
  author = {Ziang Zheng},
  title = {beyondAMP: One step unify IsaacLab with AMP.},
  url = {https://github.com/Renforce-Dynamics/beyondAMP},
  year = {2025}
}
```
