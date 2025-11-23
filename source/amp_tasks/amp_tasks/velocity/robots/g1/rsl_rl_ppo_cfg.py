# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from beyondAMP.isaaclab.rsl_rl.configs.rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from beyondAMP.isaaclab.rsl_rl.configs.amp_cfg import MotionDatasetCfg, AMPObsBaiscCfg, AMPPPOAlgorithmCfg, AMPRunnerCfg

from robotlib.robot_keys.g1_29d import g1_key_body_names, g1_anchor_name

from beyondAMP.obs_groups import AMPObsBaiscTerms

@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "g1_loco"  # same as task name
    run_name = "cliped_with_lin"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1FlatAMPRunnerCfg(AMPRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = "g1_loco"
    run_name = "amp"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = AMPPPOAlgorithmCfg(
        class_name="AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    amp_data = MotionDatasetCfg(
        motion_files=[
            "data/datasets/MocapG1Full/LAFAN/walk1_subject1.npz",
        ],
        body_names = g1_key_body_names,
        amp_obs_terms = AMPObsBaiscTerms,
        anchor_name=g1_anchor_name
    )
    amp_discr_hidden_dims = [256, 256]
    amp_reward_coef = 0.5
    amp_task_reward_lerp = 0.3
