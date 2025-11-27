from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", action="store_true", help="Whether to resume from a checkpoint.")
    # arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace, rsl_rl_cfg=None) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    if rsl_rl_cfg is None:
        # load the default configuration
        rsl_rl_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")

    # override the default configuration with CLI arguments
    if args_cli.experiment_name is not None:
        rsl_rl_cfg.experiment_name = args_cli.experiment_name
    if args_cli.seed is not None:
        rsl_rl_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        rsl_rl_cfg.resume = args_cli.resume
    # if args_cli.load_run is not None:
    #     rsl_rl_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        rsl_rl_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        rsl_rl_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        rsl_rl_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if rsl_rl_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        rsl_rl_cfg.wandb_project = args_cli.log_project_name
        rsl_rl_cfg.neptune_project = args_cli.log_project_name

    return rsl_rl_cfg

import yaml
import os
from datetime import datetime
import pickle

def update_object_from_dict(obj, data):
    for key, value in data.items():
        if hasattr(obj, key):
            if isinstance(value, dict):
                nested_obj = getattr(obj, key)
                update_object_from_dict(nested_obj, value)
            else:
                setattr(obj, key, value)
        else:
            print(f"Warning: Object has no attribute '{key}'")
    return obj

def update_object_from_yaml(obj, file_path:str):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        obj = update_object_from_dict(obj, data)
    return obj

def make_cfgs(args_cli, parse_env_cfg, runner_cfg=None):
    """Train with RSL-RL agent."""

    task_name = args_cli.task

    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)

    agent_cfg: RslRlOnPolicyRunnerCfg = parse_rsl_rl_cfg(task_name, args_cli, runner_cfg)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    agent_cfg.seed = args_cli.seed

    return task_name, env_cfg, agent_cfg, log_dir

def load_cfgs(args_cli, modified=False):
    """
    Note that if modified, this will update the saved pickle file.
    """
    task_dir = args_cli.replicate
    task_dir = os.path.abspath(task_dir)
    log_root_path = os.path.dirname(task_dir)
    with open(os.path.join(task_dir, "params", "args.pkl"), "rb") as f:
        args_old = pickle.load(f)
        task_name = args_old.task

    with open(os.path.join(task_dir, "params", "env.pkl"), "rb") as f:
        env_cfg = pickle.load(f)
        # Do not modify the env, env file will have function handles.
        # if modified: env_cfg = update_object_from_yaml(env_cfg, os.path.join(task_dir, "params", "env.yaml"))

    with open(os.path.join(task_dir, "params", "agent.pkl"), "rb") as f:
        agent_cfg: RslRlOnPolicyRunnerCfg = pickle.load(f)
        if modified: agent_cfg = update_object_from_yaml(agent_cfg, os.path.join(task_dir, "params", "agent.yaml"))

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    return task_name, env_cfg, agent_cfg, log_dir


def prepare_wrapper(env, args_cli, agent_cfg) -> Tuple[RslRlVecEnvWrapper, callable, dict]:
    print("Using main branch.")
    from beyondAMP.isaaclab.rsl_rl import AMPEnvWrapper

    env = AMPEnvWrapper(env, motion_dataset=getattr(agent_cfg, "amp_data", None))
    learn_cfg = {
        "num_learning_iterations": agent_cfg.max_iterations,
        "init_at_random_ep_len": True
    }
    
    if hasattr(agent_cfg, "runner_type"):
        func_runner = agent_cfg.runner_type
        if not callable(func_runner):
            func_runner = eval(func_runner)
    else:
        from rsl_rl_amp.runners import OnPolicyRunner
        func_runner = OnPolicyRunner

    return env, func_runner, learn_cfg


def dump_pickle(fpath, obj):
    with open(fpath, "wb") as f:
        pickle.dump(obj, f)        

