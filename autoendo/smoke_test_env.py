"""
smoke_test_env.py

Minimal Sofa env smoke test: build one headless env, step with random actions,
and print a few observations/rewards to help isolate segfaults outside SB3.
"""

from __future__ import annotations

import argparse
import numpy as np

from autoendo.rl_env import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50, help="Number of random steps to run.")
    parser.add_argument("--model", type=str, default=None, help="Optional fixed VMR model id.")
    parser.add_argument("--insert-vessel", type=str, default=None, help="Optional fixed insertion vessel.")
    parser.add_argument("--vmr-pool", type=str, default=None, help="Override VMR pool (defaults to data/vmr).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sim-dt", type=float, default=0.02)
    parser.add_argument("--image-frequency", type=float, default=5.0)
    parser.add_argument("--beam-scale", type=float, default=1.0)
    parser.add_argument("--no-mesh-check", action="store_true")
    parser.add_argument("--branch-radius", type=float, default=5.0)
    parser.add_argument("--direction-delta", type=int, default=2)
    parser.add_argument("--insert-idx", type=int, default=1)
    parser.add_argument("--action-scale", type=float, default=0.2, help="Scale factor applied to sampled actions.")
    parser.add_argument("--zero-actions", action="store_true", help="Use zero actions (no motion) to test stability.")
    parser.add_argument("--use-visualisation", action="store_true", help="Use SofaPygame visualisation (set SDL_VIDEODRIVER=dummy for headless).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        randomize_insertion_per_episode=True,
        use_visualisation=args.use_visualisation,
        noop_action=False
    )
    obs, _ = env.reset()
    print("[smoke] action_space", env.action_space)
    print("[smoke] obs shape", np.array(obs).shape)
    for step in range(args.steps):
        if args.zero_actions:
            action = np.zeros_like(env.action_space.low)
        else:
            raw = env.action_space.sample()
            action = np.clip(raw * args.action_scale, env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, info = env.step(action)
        if step % 10 == 0:
            print(f"[smoke] step={step} reward={reward} term={terminated} trunc={truncated}")
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()


if __name__ == "__main__":
    main()
