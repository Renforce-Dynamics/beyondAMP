"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import pickle
import tqdm
from isaaclab import __version__ as omni_isaac_lab_version
from isaaclab.app import AppLauncher

# local imports
import argtool as rsl_arg_cli  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
if omni_isaac_lab_version < "0.21.0":
    parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--target", type=str, default=None, help="If use, direct point to the target ckpt")

parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# parser.add_argument("--run", type=str, default=".*", help="Name of the run.")
# parser.add_argument("--ckpt", type=str, default=".*", help="Name of the ckpt.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument("--length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--rldevice", type=str, default="cuda:0", help="Device for rl")

parser.add_argument("--collect", action="store_true", default=False, help="Record data during playing.")
parser.add_argument("--web", action="store_true", default=False, help="Web videos during playing.")
parser.add_argument("--local", action="store_true", default=False, help="Using asset in local buffer")

parser.add_argument("--determine",action="store_true", default=False, help="Clear reset terms" )

# append RSL-RL cli arguments
rsl_arg_cli.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video or args_cli.web:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import os
import torch

import amp_tasks
# import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.utils.dict import print_dict

# Import extensions to set up environment tasks
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from rsl_rl_amp.runners import OnPolicyRunner
from beyondAMP.isaaclab.rsl_rl.exporter import export_policy_as_jit, export_policy_as_onnx

def main():
    """Play with RSL-RL agent. base branch"""
    task_name = args_cli.task
    if args_cli.target is None:
        raise ValueError("Please using the target specified way.")
    else:
        resume_path = os.path.abspath(args_cli.target)
        log_root_path = os.path.dirname(os.path.dirname(resume_path))
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        log_dir = os.path.dirname(resume_path)
        run_path = os.path.dirname(resume_path)
    
    if args_cli.collect:
        sample_dir = os.path.join(run_path, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        output_file = os.path.join(sample_dir, "total_data.pkl")
    
    with open(os.path.join(run_path, "params", "agent.pkl"), "rb") as f:
        agent_cfg = pickle.load(f)
        
    if task_name is None:
        assert os.path.exists(os.path.join(run_path, "params", "args.pkl")), "No task specified."
        with open(os.path.join(run_path, "params", "args.pkl"), "rb") as f:
            args_old = pickle.load(f)
        task_name = args_old.task

        with open(os.path.join(run_path, "params", "env.pkl"), "rb") as f:
            env_cfg = pickle.load(f)
    else:
        env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")

    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed

    from isaaclab.envs.common import ViewerCfg
    
    
    env_cfg.viewer = ViewerCfg(
        eye = (4.0, 4.0, 4.0),
        # eye = (0.0, 0.0, 10.0),
        lookat = (0.0, 0.0, 0.0),
        env_index = 20,
        origin_type = "asset_root",
        # origin_type = "env",
        asset_name = "robot",
    )
    
    env_cfg.curriculum = None
    # if args_cli.determine:
    #     set_determine_reset(env_cfg)
    
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.8, 1.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    
    # env_cfg.commands.base_velocity.debug_vis = False
    # env_cfg.scene.height_scanner.debug_vis = False

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    render_mode = "rgb_array" if args_cli.video or args_cli.web else None
    # create isaac environment
    env = gym.make(task_name, cfg=env_cfg, render_mode=render_mode)
    
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.length if args_cli.length > 0 else 0,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        # print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.rldevice

    env, func_runner, learn_cfg = rsl_arg_cli.prepare_wrapper(env, args_cli, agent_cfg)

    runner:OnPolicyRunner = func_runner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # reset environment
    obs = env.get_observations()

    # export policy
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_model_dir, exist_ok=True)
    torch.save(runner.alg.actor_critic, os.path.join(export_model_dir, "policy.pth"))
    export_policy_as_onnx(runner.alg.actor_critic, export_model_dir, filename="policy.onnx")
    export_policy_as_jit(runner.alg.actor_critic, None, export_model_dir, filename="policy.pt")
    # export_policy_as_jit(runner.alg.policy, runner.alg.policy.actor_obs_normalizer, export_model_dir, filename="policy.pt")

    print(f"[INFO]: Saving policy to: {export_model_dir}")

    pbar = tqdm.tqdm(range(args_cli.length)) if args_cli.length>0  else tqdm.tqdm()
    
    step = 0
    try:
        # simulate environment
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, rewards, dones, infos = env.step(actions, not_amp=True)

            step += 1
            pbar.update() 
            if args_cli.collect:
                # trans = [obs.data, actions.data, rewards.data, dones.data, ppo_runner.alg.actor_critic.laction.data]
                trans = [obs.data, actions.data, rewards.data, dones.data]
                with open(output_file, 'ab') as f:
                    pickle.dump(trans, f)
            if args_cli.length > 0 and args_cli.length < step:
                break

    except KeyboardInterrupt:
        pass

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
