"""
rl_env.py

Utility functions to build the stEVE VMR endovascular environment for RL training.
Provides single and vectorised env factories (headless) that mirror autonomy_train.py
setup, including random VMR/branch selection and flattened observations for SB3.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional
import os
import random
import warnings
import heapq

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

import eve
from eve.visualisation import VisualisationDummy, SofaPygame
from eve.observation.observation import Observation as EveObservation
from gymnasium import Wrapper as GymWrapper

from .autonomy_test import (
    _default_vmr_pool,
    ensure_model_available,
    list_branch_names,
)


class LocalCenterlinePatch(EveObservation):
    """Local centerline patch around the tip, plus radius and target direction."""

    def __init__(
        self,
        intervention,
        vessel_tree,
        k: int = 5,
        radius_hint: float = 0.0,
        name: str = "local_patch",
    ) -> None:
        self.name = name
        self.intervention = intervention
        self.vessel_tree = vessel_tree
        self.k = k
        self.radius_hint = radius_hint
        self._printed_error = False
        self.obs = None

    @property
    def space(self) -> gym.Space:
        # k points * 3 + 1 radius + 3 target dir
        size = self.k * 3 + 4
        low = np.full((size,), -1e6, dtype=np.float32)
        high = np.full((size,), 1e6, dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _tip_and_tangent(self):
        tracking = self.intervention.fluoroscopy.tracking3d
        tip = np.array(tracking[0], dtype=np.float32)
        if len(tracking) > 1:
            tangent = np.array(tracking[0] - tracking[1], dtype=np.float32)
        else:
            tangent = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        norm = np.linalg.norm(tangent)
        if norm < 1e-6:
            tangent = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            tangent = tangent / norm
        return tip, tangent

    def _nearest_points(self, tip: np.ndarray) -> np.ndarray:
        if getattr(self.vessel_tree, "centerline_coordinates", None) is not None:
            coords = self.vessel_tree.centerline_coordinates
        else:
            coords = np.concatenate([b.coordinates for b in self.vessel_tree.branches])
        dists = np.linalg.norm(coords - tip, axis=1)
        idx = int(np.argmin(dists))
        half = self.k // 2
        start = max(0, idx - half)
        end = min(coords.shape[0], start + self.k)
        start = max(0, end - self.k)
        patch = coords[start:end]
        if patch.shape[0] < self.k:
            pad = np.repeat(patch[-1][None, :], self.k - patch.shape[0], axis=0)
            patch = np.vstack([patch, pad])
        return patch

    def step(self) -> None:
        try:
            tip, _ = self._tip_and_tangent()
            patch = self._nearest_points(tip) - tip  # translate to tip frame
            target = np.array(self.intervention.target.coordinates3d, dtype=np.float32)
            tvec = target - tip
            tnorm = np.linalg.norm(tvec)
            if tnorm > 1e-6:
                tdir = tvec / tnorm
            else:
                tdir = np.zeros(3, dtype=np.float32)
            radius = float(self.radius_hint)
            obs_vec = np.concatenate([patch.flatten(), np.array([radius], dtype=np.float32), tdir])
            self.obs = obs_vec.astype(np.float32)
        except Exception as exc:  # pragma: no cover
            if not self._printed_error:
                print(f"[local_patch] error computing patch: {exc}")
                self._printed_error = True
            size = self.k * 3 + 4
            self.obs = np.zeros((size,), dtype=np.float32)

    def reset(self, episode_nr: int = 0) -> None:
        self._printed_error = False
        self.step()


class RewardComponentWrapper(GymWrapper):
    """Attach reward component values to info for logging."""

    def __init__(self, env, components: dict[str, any]):
        super().__init__(env)
        self._components = components

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            info = dict(info)
            info["reward_components"] = {k: getattr(c, "reward", None) for k, c in self._components.items()}
        except Exception as exc:  # pragma: no cover
            print(f"[reward_components] failed to attach: {exc}")
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def _build_env(
    *,
    model: Optional[str],
    vmr_pool,
    insert_vessel: Optional[str],
    insert_idx: int,
    direction_delta: int,
    branch_radius: float,
    rotate: Optional[Iterable[float]],
    no_mesh_check: bool,
    target_branches: Optional[Iterable[str]],
    beam_scale: float,
    sim_dt: float,
    image_frequency: float,
    randomize_insertion_per_episode: bool,
    use_visualisation: bool = False,
    patch_size: int = 5,
) -> gym.Env:
    def _configure_headless_display() -> None:
        """Set SDL env for headless SofaPygame if a display is not available."""
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    vmr_pool = vmr_pool or _default_vmr_pool()
    available_models = ensure_models_cached(vmr_pool)
    chosen_model = model or random.choice(available_models)
    model_dir = ensure_model_available(chosen_model, vmr_pool)

    branch_names = list_branch_names(model_dir)
    insertion_vessel = (insert_vessel or random.choice(branch_names)).lower()
    if insertion_vessel not in branch_names:
        raise ValueError(f"Insertion vessel '{insertion_vessel}' not present. Candidates: {branch_names}")
    target_branch_list = list(target_branches) if target_branches else [b for b in branch_names if b != insertion_vessel]
    if not target_branch_list:
        target_branch_list = branch_names

    vessel_tree = eve.intervention.vesseltree.VMR(
        model=chosen_model,
        insertion_vessel_name=insertion_vessel,
        insertion_point_idx=insert_idx,
        insertion_direction_idx_diff=direction_delta,
        approx_branch_radii=branch_radius,
        rotate_yzx_deg=tuple(rotate) if rotate else None,
        check_if_points_in_mesh=not no_mesh_check,
        randomize_insertion_each_reset=randomize_insertion_per_episode,
    )

    device = eve.intervention.device.JShaped()
    if beam_scale != 1.0:
        sofa_device = device.sofa_device
        density = np.asarray(sofa_device.density_of_beams, dtype=float)
        scaled = np.maximum(1, np.round(density * beam_scale).astype(int))
        sofa_device.density_of_beams = tuple(int(v) for v in scaled)
        if sofa_device.num_edges_collis is not None:
            collis = np.asarray(sofa_device.num_edges_collis, dtype=float)
            sofa_device.num_edges_collis = tuple(int(max(1, round(c * beam_scale))) for c in collis)

    simulation = eve.intervention.simulation.SofaBeamAdapter(friction=0.1, dt_simulation=sim_dt)
    fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
        simulation=simulation,
        vessel_tree=vessel_tree,
        image_frequency=image_frequency,
        image_rot_zx=[20, 5],
    )
    target = eve.intervention.target.CenterlineRandom(
        vessel_tree=vessel_tree,
        fluoroscopy=fluoroscopy,
        threshold=5,
        branches=target_branch_list,
    )
    intervention = eve.intervention.MonoPlaneStatic(
        vessel_tree=vessel_tree,
        devices=[device],
        simulation=simulation,
        fluoroscopy=fluoroscopy,
        target=target,
    )
    start = eve.start.MaxDeviceLength(intervention=intervention, max_length=1000)
    pathfinder = eve.pathfinder.BruteForceBFS(intervention=intervention)
    _debug_branch_paths(vessel_tree, insertion_vessel, target_branch_list)

    position = eve.observation.Tracking2D(intervention=intervention, n_points=5)
    position = eve.observation.wrapper.NormalizeTracking2DEpisode(position, intervention)
    target_state = eve.observation.Target2D(intervention=intervention)
    target_state = eve.observation.wrapper.NormalizeTracking2DEpisode(target_state, intervention)
    rotation = eve.observation.Rotations(intervention=intervention)
    local_patch = LocalCenterlinePatch(
        intervention=intervention,
        vessel_tree=vessel_tree,
        k=patch_size,
        radius_hint=branch_radius,
    )
    obs = eve.observation.ObsDict(
        {"position": position, "target": target_state, "rotation": rotation, "local_patch": local_patch}
    )

    target_reward = eve.reward.TargetReached(intervention=intervention, factor=2.0)
    path_delta = eve.reward.PathLengthDelta(pathfinder=pathfinder, factor=0.05)
    tip_progress = eve.reward.TipToTargetDistDelta(
        factor=0.1, intervention=intervention, interim_target=None
    )
    reward = eve.reward.Combination([target_reward, path_delta, tip_progress])

    terminal = eve.terminal.TargetReached(intervention=intervention)
    truncation = eve.truncation.MaxSteps(200)

    if use_visualisation:
        _configure_headless_display()

    visualisation = SofaPygame(intervention=intervention) if use_visualisation else VisualisationDummy()

    env = eve.Env(
        intervention=intervention,
        observation=obs,
        reward=reward,
        terminal=terminal,
        truncation=truncation,
        visualisation=visualisation,
        start=start,
        pathfinder=pathfinder,
    )
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.ClipAction(env)
    env = RewardComponentWrapper(env, components={"target": target_reward, "path": path_delta, "tip": tip_progress})
    limits = np.abs(np.array(intervention.velocity_limits, dtype=np.float32)).reshape(-1)
    env.action_space = gym.spaces.Box(low=-limits, high=limits, dtype=np.float32)
    print(f"[Env] obs_dim={env.observation_space.shape[0]} action_dim={env.action_space.shape[0]}")

    return env


def ensure_models_cached(vmr_pool) -> list[str]:
    available = []
    vmr_pool = vmr_pool or _default_vmr_pool()
    if vmr_pool.is_dir():
        for entry in vmr_pool.iterdir():
            if entry.name.startswith("."):
                continue
            if entry.is_file() and entry.suffix.lower() == ".zip":
                available.append(entry.stem)
            elif entry.is_dir():
                available.append(entry.name)
    if not available:
        raise RuntimeError(f"No VMR models discovered under {vmr_pool}")
    return sorted(available)


def make_env(
    *,
    model: Optional[str] = None,
    vmr_pool=_default_vmr_pool(),
    insert_vessel: Optional[str] = None,
    insert_idx: int = 1,
    direction_delta: int = 2,
    branch_radius: float = 5.0,
    rotate: Optional[Iterable[float]] = None,
    no_mesh_check: bool = False,
    target_branches: Optional[Iterable[str]] = None,
    beam_scale: float = 1.0,
    sim_dt: float = 0.02,
    image_frequency: float = 5.0,
    seed: Optional[int] = None,
    randomize_insertion_per_episode: bool = True,
    use_visualisation: bool = False,
    noop_action: bool = False,
    patch_size: int = 5,
) -> gym.Env:
    env = _build_env(
        model=model,
        vmr_pool=vmr_pool,
        insert_vessel=insert_vessel,
        insert_idx=insert_idx,
        direction_delta=direction_delta,
        branch_radius=branch_radius,
        rotate=rotate,
        no_mesh_check=no_mesh_check,
        target_branches=target_branches,
        beam_scale=beam_scale,
        sim_dt=sim_dt,
        image_frequency=image_frequency,
        randomize_insertion_per_episode=randomize_insertion_per_episode,
        use_visualisation=use_visualisation,
        patch_size=patch_size,
    )
    if noop_action:
        # Freeze actions to zeros for debugging stability.
        env = gym.wrappers.TransformReward(env, lambda r: r)
        env = gym.wrappers.TransformObservation(env, lambda o: o, observation_space=env.observation_space)
        orig_step = env.step

        def step_zero(_action):
            return orig_step(np.zeros_like(env.action_space.low))

        env.step = step_zero  # type: ignore
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_vec_env(
    num_envs: int = 1,
    *,
    model: Optional[str] = None,
    vmr_pool=_default_vmr_pool(),
    insert_vessel: Optional[str] = None,
    insert_idx: int = 1,
    direction_delta: int = 2,
    branch_radius: float = 5.0,
    rotate: Optional[Iterable[float]] = None,
    no_mesh_check: bool = False,
    target_branches: Optional[Iterable[str]] = None,
    beam_scale: float = 1.0,
    sim_dt: float = 0.02,
    image_frequency: float = 5.0,
    seed: Optional[int] = None,
    use_subproc_envs: bool = False, # auto-detect if num_envs>1
    randomize_insertion_per_episode: bool = True,
    use_visualisation: bool = False,
    patch_size: int = 10,
) -> VecEnv:
    def make_one(rank: int, base_seed: Optional[int]) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            env_seed = None if base_seed is None else base_seed + rank
            env = make_env(
                model=model,
                vmr_pool=vmr_pool,
                insert_vessel=insert_vessel,
                insert_idx=insert_idx,
                direction_delta=direction_delta,
                branch_radius=branch_radius,
                rotate=rotate,
                no_mesh_check=no_mesh_check,
                target_branches=target_branches,
                beam_scale=beam_scale,
                sim_dt=sim_dt,
                image_frequency=image_frequency,
                seed=env_seed,
                randomize_insertion_per_episode=randomize_insertion_per_episode,
                use_visualisation=use_visualisation,
                patch_size=patch_size,
            )
            return env

        return _init

    env_fns = [make_one(i, seed) for i in range(num_envs)]
    if use_subproc_envs or num_envs > 1:
        return SubprocVecEnv(env_fns, start_method="spawn")
    return DummyVecEnv(env_fns)


def _debug_branch_paths(vessel_tree, insertion_vessel: str, target_branches: list[str]) -> None:
    """Compute a simple branch graph and Dijkstra reachability for debugging."""

    def build_graph():
        adj: dict[str, list[tuple[str, float]]] = {}
        branches = vessel_tree.branches
        if branches is None:
            return {}
        name_to_branch = {b.name: b for b in branches}
        for b in branches:
            adj[b.name] = []
        branching_points = getattr(vessel_tree, "branching_points", None) or []
        for bp in branching_points:
            conns = list(bp.connections)
            for i in range(len(conns)):
                for j in range(i + 1, len(conns)):
                    a = conns[i].name
                    b = conns[j].name
                    la = name_to_branch[a].length
                    lb = name_to_branch[b].length
                    w = float((la + lb) / 2.0)
                    adj[a].append((b, w))
                    adj[b].append((a, w))
        return adj

    try:
        graph = build_graph()
        if not graph or insertion_vessel not in graph:
            print(f"[paths] graph missing insertion branch or empty (has={len(graph)})")
            return
        dist: dict[str, float] = {insertion_vessel: 0.0}
        heap = [(0.0, insertion_vessel)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, w in graph.get(u, []):
                nd = d + w
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        reachable_targets = [t for t in target_branches if t in dist]
        unreachable = [t for t in target_branches if t not in dist]
        n_edges = sum(len(v) for v in graph.values()) // 2
        print(
            f"[paths] nodes={len(graph)} edges={n_edges} reachable={len(dist)} "
            f"targets_reachable={len(reachable_targets)} unreachable={unreachable}"
        )
    except Exception as exc:  # pragma: no cover
        print(f"[paths] error computing graph/dijkstra: {exc}")
