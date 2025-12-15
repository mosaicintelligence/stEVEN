"""
train_sb3.py

SB3 training harness for stEVE VMR navigation with PPO/SAC.
Builds headless environments from rl_env.py (random VMR/branch by default),
supports vectorised rollout, TensorBoard logging, and checkpoint saving.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor, VecFrameStack
import datetime
from autoendo.rl_env import make_vec_env, make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vmr-pool", type=Path, default=None, help="Override VMR pool (defaults to data/vmr)")
    parser.add_argument("--model", type=str, default=None, help="Specific VMR model id. If omitted, random per env.")
    parser.add_argument("--insert-vessel", type=str, default=None, help="Optional fixed insertion vessel name.")
    parser.add_argument("--beam-scale", type=float, default=1.0, help="Scale procedural device beams.")
    parser.add_argument("--sim-dt", type=float, default=0.02, help="Simulation timestep seconds.")
    parser.add_argument("--image-frequency", type=float, default=5.0, help="Fluoro/image frequency in Hz.")
    parser.add_argument("--branch-radius", type=float, default=5.0, help="Approx branch radius for branching detection.")
    parser.add_argument("--direction-delta", type=int, default=2, help="Offset for insertion direction index.")
    parser.add_argument("--insert-idx", type=int, default=1, help="Insertion point index on branch.")
    parser.add_argument("--no-mesh-check", action="store_true", help="Disable mesh containment filtering.")
    parser.add_argument("--frame-stack", type=int, default=1, help="Number of stacked observations for temporal context.")
    parser.add_argument(
        "--recurrent",
        action="store_true",
        help="Use Recurrent PPO (LSTM) for temporal context (PPO only).",
    )
    parser.add_argument("--lstm-hidden-size", type=int, default=256, help="Hidden size for LSTM when recurrent PPO is used.")
    parser.add_argument("--lstm-n-layers", type=int, default=1, help="Number of LSTM layers for recurrent PPO.")
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/autoendo/sb3_logs"))
    parser.add_argument("--save-path", type=Path, default=Path("outputs/autoendo/sb3_model"))
    parser.add_argument("--no-log", action="store_true", help="Disable TensorBoard logging under log-dir/tb.")
    parser.add_argument("--eval-episodes", type=int, default=8, help="Episodes for post-train evaluation.")
    parser.add_argument("--device", type=str, default="auto", help="Device for torch (auto/cpu/cuda).")
    parser.add_argument("--subproc-envs", action="store_true", help="Use SubprocVecEnv (process isolation) even for 1 env.")
    parser.add_argument("--use-visualisation", action="store_true", help="Use SofaPygame visualisation (set SDL_VIDEODRIVER=dummy for headless).")
    return parser.parse_args()


def build_envs(args: argparse.Namespace):
    env_kwargs = dict(
        model=args.model,
        vmr_pool=args.vmr_pool,
        insert_vessel=args.insert_vessel,
        insert_idx=args.insert_idx,
        direction_delta=args.direction_delta,
        branch_radius=args.branch_radius,
        no_mesh_check=args.no_mesh_check,
        beam_scale=args.beam_scale,
        sim_dt=args.sim_dt,
        image_frequency=args.image_frequency,
        use_visualisation=args.use_visualisation,
    )
    vec_env = make_vec_env(
        num_envs=args.num_envs,
        seed=args.seed,
        use_subproc_envs=args.subproc_envs,
        **env_kwargs,
    )
    if args.frame_stack and args.frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=args.frame_stack)
    vec_env = VecMonitor(vec_env, filename=str(args.log_dir / "monitor"))

    # Ensure finite action bounds on the VecEnv (work around any wrapper issues).
    sample_env = make_env(seed=args.seed, **env_kwargs)
    base = sample_env.unwrapped
    limits = np.abs(np.array(base.intervention.velocity_limits, dtype=np.float32)).reshape(-1)
    finite_box = gym.spaces.Box(low=-limits, high=limits, dtype=np.float32)
    vec_env.action_space = finite_box
    print(f"[Env] action space low: {finite_box.low}, high: {finite_box.high}")
    sample_env.close()

    eval_env = make_vec_env(
        num_envs=1,
        seed=args.seed + 10_000,
        use_subproc_envs=args.subproc_envs,
        **env_kwargs,
    )
    if args.frame_stack and args.frame_stack > 1:
        eval_env = VecFrameStack(eval_env, n_stack=args.frame_stack)
    eval_env = VecMonitor(eval_env, filename=str(args.log_dir / "monitor_eval"))
    return vec_env, eval_env


def make_model(args: argparse.Namespace, env: gym.Env):
    # Validate action space finiteness for off-policy algos like SAC.
    if not hasattr(env, "action_space"):
        raise ValueError("Vector env missing action_space.")
    low = env.action_space.low
    high = env.action_space.high
    if not (low is not None and high is not None):
        raise ValueError(f"Action space has undefined bounds: low={low}, high={high}")
    if not (low.shape == high.shape):
        raise ValueError(f"Action space low/high shape mismatch: {low.shape} vs {high.shape}")
    if not (np.isfinite(low).all() and np.isfinite(high).all()):
        raise ValueError(f"Non-finite action bounds detected: low={low}, high={high}")

    common_kwargs = dict(
        env=env,
        seed=args.seed,
        device=args.device,
        verbose=1,
        tensorboard_log=None if args.no_log else (args.log_dir),
    )
    if args.algo == "ppo":
        if args.recurrent:
            try:
                from stable_baselines3 import RecurrentPPO  # type: ignore
            except ImportError:
                try:
                    from sb3_contrib import RecurrentPPO  # type: ignore
                except ImportError as exc:  # pragma: no cover
                    raise RuntimeError("RecurrentPPO not available; install sb3-contrib or SB3>=2.3.") from exc
            policy_kwargs = dict(
                lstm_hidden_size=args.lstm_hidden_size,
                n_lstm_layers=args.lstm_n_layers,
            )
            return RecurrentPPO(
                "MlpLstmPolicy",
                **common_kwargs,
                n_steps=max(128, 2048 // args.num_envs),
                batch_size=64,
                gae_lambda=0.95,
                policy_kwargs=policy_kwargs,
            )
        return PPO("MlpPolicy", **common_kwargs, n_steps=2048 // args.num_envs, batch_size=64, gae_lambda=0.95)
    if args.recurrent:
        raise ValueError("Recurrent flag is only supported for PPO at the moment.")
    return SAC(
        "MlpPolicy",
        **common_kwargs,
        buffer_size=500_000,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
    )


def main() -> None:
    args = parse_args()
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    args.log_dir = args.log_dir / f"{args.algo}_{datetime_str}"
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    args.save_path = args.save_path.parent / f"{args.save_path.stem}_{args.algo}_{datetime_str}{args.save_path.suffix}"

    vec_env, eval_env = build_envs(args)
    model = make_model(args, vec_env)

    checkpoint_cb = CheckpointCallback(save_freq=5000, save_path=args.log_dir, name_prefix=f"{args.algo}_ckpt")
    model.learn(total_timesteps=args.total_steps, callback=checkpoint_cb, progress_bar=True)
    model.save(str(args.save_path))

    if args.eval_episodes > 0:
        mean_rew, std_rew = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes, deterministic=True)
        print(f"[eval] mean_reward={mean_rew:.3f} +- {std_rew:.3f}")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
