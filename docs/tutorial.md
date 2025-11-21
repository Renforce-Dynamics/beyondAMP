# BeyondAMP Tutorial

This document introduces how to use **BeyondAMP** in your custom Isaac Lab tasks.  
It also summarizes the essential concepts of **Adversarial Motion Priors (AMP)** and explains how BeyondAMP integrates AMP-style rewards into RSL-RL workflows.

---

## 1. Background: What AMP Does

AMP provides **style rewards** by comparing the agent’s behavior to **prior motion data**.  
It trains a **discriminator** to distinguish:
- real motion clips sampled from your dataset vs.  
- agent-generated motion traces during RL.

If the discriminator cannot tell them apart, the agent is behaving in a “motion-prior-consistent” manner.

In practice:
1. You define **AMP observation terms** (joint pos/vel, body orientation, velocities, etc.).  
2. The environment outputs AMP observations every step.  
3. Motion data is converted into AMP observations as well.  
4. The discriminator computes reward based on similarity.

BeyondAMP extends this system with clean Isaac Lab wrappers and flexible observation configurations.

---

## 2. Adding BeyondAMP to Your Task

### 2.1 AMP Observation Groups

BeyondAMP provides several pre-defined AMP observation groups in  
`source/beyondAMP/beyondAMP/obs_groups.py`.

Example:

```python
@configclass
class AMPObsBaiscCfg(AMPObsGrpCfg):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)

AMPObsBaiscTerms = ["joint_pos", "joint_vel"]


@configclass
class AMPObsSoftTrackCfg(AMPObsGrpCfg):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    body_quat_w = ObsTerm(func=mdp.body_quat_w)
    body_lin_vel_w = ObsTerm(func=mdp.body_lin_vel_w)
    body_ang_vel_w = ObsTerm(func=mdp.body_ang_vel_w)

AMPObsSoftTrackTerms = [
    "joint_pos", "joint_vel",
    "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"
]
````

**Key idea:**

* These configs determine *what* goes into the AMP observation vector.
* The same term set must be used by:

  * the environment’s AMP observation generator, and
  * the motion dataset converter.

Basic AMP (joint pos/vel only) provides weak but highly general motion priors.
Soft-tracking AMP adds pose/velocity terms in world frame, useful for tasks requiring trajectory tracking.

---

## 3. Injecting AMP Into Your Environment

In Isaac Lab’s manager-style task config, simply add an AMP observation group.

Example (from `g1/flat_env_cfg.py`):

```python
@configclass
class G1FlatEnvBasicCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.amp = AMPObsBaiscCfg()
```

Once attached, the environment will automatically produce AMP observations at every step.

For body tracking terms, please ensure your amp body names consistent with dataset names: 

```
@configclass
class G1FlatEnvSoftTrackCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.amp = \
            AMPObsSoftTrackCfg().adjust_key_body_indexes(
                ["body_quat_w", "body_lin_vel_w", "body_ang_vel_w"],
                g1_key_body_names
                )
```

## 4. AMP Training Configuration

BeyondAMP builds its training pipeline on **RSL-RL**, using a custom `AMPPPO` algorithm built on top of PPO.

Example (from `rsl_rl_ppo_cfg.py`):

```python
algorithm = AMPPPOAlgorithmCfg(
    class_name="AMPPPO",
    ...
)

amp_data = MotionDatasetCfg(
    motion_files=[
        "data/demo/punch_000.npz"
    ],
    body_names=g1_key_body_names,
    amp_obs_terms=None      # or a term list such as AMPObsBaiscTerms
)

amp_discr_hidden_dims = [256, 256]
amp_reward_coef = 0.5
amp_task_reward_lerp = 0.3
```

### 4.1 Parameter Description

| Parameter               | Meaning                             | Notes                                       |
| ----------------------- | ----------------------------------- | ------------------------------------------- |
| `motion_files`          | Paths to motion datasets (.npz)     | Each file is one trajectory or sequence set |
| `body_names`            | Body references for FK              | Must match robot asset names                |
| `amp_obs_terms`         | AMP term list                       | If `None`, use default from obs group       |
| `amp_discr_hidden_dims` | Discriminator network MLP sizes     | Larger → stronger style discrimination      |
| `amp_reward_coef`       | AMP reward scaling factor           | Normalizes discriminator reward magnitude   |
| `amp_task_reward_lerp`  | Balance between task and AMP reward | `1.0 = pure task`; `0.0 = pure AMP mimicry` |
| `class_name="AMPPPO"`   | Use BeyondAMP’s PPO extension       | Compatible with RSL-RL wrappers             |

---

## 5. Starting Training

Training is nearly identical to standard RSL-RL pipelines.

Ensure your environment is wrapped by BeyondAMP’s **AMPEnvWrapper**:

```python
from beyondAMP.isaaclab.rsl_rl import AMPEnvWrapper

env = AMPEnvWrapper(env)
```

This wrapper:

* manages AMP observation extraction,
* handles dataset sampling,
* computes discriminator rewards,
* and injects AMP reward into PPO training.

Then run training as usual with RSL-RL.

> Future versions will support **skrl** and other RL frameworks.

---

## 6. Summary

BeyondAMP provides:

* flexible AMP observation definitions,
* seamless integration with Isaac Lab’s env config system,
* RSL-RL compatible AMPPPO training,
* automatic motion-prior reward computation.

With just:

1. an AMP observation config,
2. a motion dataset,
3. and the AMPEnvWrapper,

you can augment any Isaac Lab task with strong motion-style priors.

---

If you want, I can also help you write:

* a **quickstart version**,
* a **paper-style section**,
* or a **full documentation chapter** aligned with Isaac Lab’s documentation style.

```
