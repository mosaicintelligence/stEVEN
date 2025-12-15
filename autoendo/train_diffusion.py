"""
train_diffusion.py

Behavior-cloning style training loop for the diffusion action policy.
Collects (obs, action) pairs from a random or SB3 policy, trains the
DiffusionPolicy, and evaluates it in-env for quick sanity checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO, SAC

from .policy_diffusion import DiffusionPolicy
from .rl_env import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-steps", type=int, default=25_000, help="Transitions to collect for training.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--save-path", type=Path, default=Path("outputs/autoendo/diffusion_policy.pt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--behavior-path", type=Path, default=None, help="Optional SB3 policy to collect data.")
    parser.add_argument("--behavior-algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--vmr-pool", type=Path, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--insert-vessel", type=str, default=None)
    parser.add_argument("--insert-idx", type=int, default=1)
    parser.add_argument("--direction-delta", type=int, default=2)
    parser.add_argument("--branch-radius", type=float, default=5.0)
    parser.add_argument("--no-mesh-check", action="store_true")
    parser.add_argument("--beam-scale", type=float, default=1.0)
    parser.add_argument("--sim-dt", type=float, default=0.02)
    parser.add_argument("--image-frequency", type=float, default=5.0)
    return parser.parse_args()


def load_behavior_model(args: argparse.Namespace, env):
    if args.behavior_path is None:
        return None
    if args.behavior_algo == "ppo":
        return PPO.load(args.behavior_path, env=env, device=args.device)
    return SAC.load(args.behavior_path, env=env, device=args.device)


def collect_dataset(env, steps: int, behavior_model=None, seed: Optional[int] = None):
    rng = np.random.default_rng(seed)
    obs_list = []
    act_list = []
    obs, _ = env.reset(seed=seed)
    for _ in range(steps):
        if behavior_model is None:
            action = env.action_space.sample()
        else:
            action, _ = behavior_model.predict(obs, deterministic=True)
        obs_list.append(obs)
        act_list.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset(seed=rng.integers(0, 1_000_000))
    return np.stack(obs_list), np.stack(act_list)


def rollout_eval(env, policy: DiffusionPolicy, device, episodes: int = 5):
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            action = policy.sample(obs_t).squeeze(0).cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
    return returns


def main() -> None:
    args = parse_args()
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = make_env(
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
        seed=args.seed,
    )
    behavior_model = load_behavior_model(args, env)

    print("[diffusion] collecting dataset...")
    obs_np, act_np = collect_dataset(env, steps=args.rollout_steps, behavior_model=behavior_model, seed=args.seed)
    obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32)
    act_tensor = torch.as_tensor(act_np, dtype=torch.float32)

    dataset = TensorDataset(obs_tensor, act_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    policy = DiffusionPolicy(
        obs_dim=obs_tensor.shape[1],
        action_dim=act_tensor.shape[1],
        timesteps=args.diffusion_steps,
    ).to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)

    print("[diffusion] training...")
    for epoch in range(args.epochs):
        running = 0.0
        for batch_obs, batch_act in loader:
            batch_obs = batch_obs.to(device)
            batch_act = batch_act.to(device)
            optim.zero_grad()
            loss = policy.loss(batch_obs, batch_act)
            loss.backward()
            optim.step()
            running += loss.item()
        avg = running / max(1, len(loader))
        print(f"[diffusion] epoch {epoch+1}/{args.epochs} loss={avg:.4f}")

    torch.save(
        {
            "state_dict": policy.state_dict(),
            "obs_dim": obs_tensor.shape[1],
            "action_dim": act_tensor.shape[1],
            "timesteps": args.diffusion_steps,
        },
        args.save_path,
    )
    print(f"[diffusion] saved to {args.save_path}")

    if args.eval_episodes > 0:
        returns = rollout_eval(env, policy, device, episodes=args.eval_episodes)
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))
        print(f"[diffusion] eval mean_return={mean_ret:.3f} +- {std_ret:.3f}")

    env.close()


if __name__ == "__main__":
    main()
