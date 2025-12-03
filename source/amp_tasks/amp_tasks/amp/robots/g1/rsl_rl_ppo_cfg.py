from isaaclab.utils import configclass
from beyondAMP.isaaclab.rsl_rl.configs.rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from beyondAMP.isaaclab.rsl_rl.configs.amp_cfg import \
    MotionDatasetCfg, AMPObsBaiscCfg, AMPPPOAlgorithmCfg, AMPRunnerCfg, AMPPPOWeightedAlgorithmCfg

from beyondAMP.obs_groups import AMPObsBaiscTerms, AMPObsSoftTrackTerms, AMPObsHardTrackTerms

from robotlib.robot_keys.g1_29d import g1_key_body_names, g1_anchor_name

@configclass
class G1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "g1_flat_demo"
    empirical_normalization = True
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

@configclass
class G1FlatAMPRunnerCfg(AMPRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "g1_flat_demo"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = AMPPPOWeightedAlgorithmCfg(
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
        rescore_interval=100,
    )
    amp_data = MotionDatasetCfg(
        motion_files=[
            "data/demo/punch_000.npz"
        ],
        body_names = g1_key_body_names,
        anchor_name = g1_anchor_name,
        amp_obs_terms = None,
    )
    amp_discr_hidden_dims = [256, 256]
    amp_reward_coef = 0.5
    amp_task_reward_lerp = 0.3


@configclass
class G1FlatAMPBaiscCfg(G1FlatAMPRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.amp_data.amp_obs_terms = AMPObsBaiscTerms
        self.run_name = "basic"
        self.amp_task_reward_lerp = 0.05
        self.amp_reward_coef = 1.0


@configclass
class G1FlatAMPSoftTrackCfg(G1FlatAMPRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.amp_data.amp_obs_terms = AMPObsSoftTrackTerms
        self.run_name = "soft_track"
        self.amp_data.motion_files = []

@configclass
class G1FlatAMPHardTrackCfg(G1FlatAMPRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.amp_data.amp_obs_terms = AMPObsHardTrackTerms
        self.run_name = "hard_track"
        self.amp_data.motion_files = []