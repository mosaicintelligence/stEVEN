"""
autonomy_train.py

Main training environment for endovascular autonomy. Runs across VMR models
with a J-shaped wire device
setting up for RL policy training

policy inputs: 
- vascular map observation: 
    - centerlines, insertion point/direction, target point
    - possibly: 2D fluoroscopy images (simulated), or 3D volumetric manifold (SDF etc)
    - stEVE standard observations: 2D tracking points, target position, rotations
    - pathfinder for path length reward (BruteForceBFS)
- state space: instrument point poses (multiple points along wire)
- action space: translation and rotation at base of wire
- reward: reaching target, path length penalty

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import zipfile
from time import perf_counter
import math
import functools

import numpy as np
import pygame
import random
import gymnasium as gym
import torch

import eve
from eve.visualisation.sofapygame import SofaPygame


LOGGER = logging.getLogger(__name__)


def _default_vmr_pool() -> Path:
    """Return the repo-local ``data/vmr`` directory."""

    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "vmr"


def _ensure_internal_vmr_dir() -> Path:
    """Return ``stEVEN/.data/vmr`` creating it if missing."""

    steve_root = Path(__file__).resolve().parents[1]
    data_root = steve_root / ".data"
    data_root.mkdir(exist_ok=True)
    vmr_dir = data_root / "vmr"
    vmr_dir.mkdir(exist_ok=True)
    return vmr_dir


def _copy_directory(src: Path, dst: Path) -> None:
    """Copy ``src`` folder into ``dst`` when it does not yet exist."""

    if dst.exists():
        return
    shutil.copytree(src, dst)


def _extract_zip(zip_path: Path, target_root: Path) -> Path:
    """Extract ``zip_path`` into ``target_root`` and return the model folder."""

    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.namelist() if not m.startswith("__MACOSX")]
        archive.extractall(target_root, members)

    root_candidates = {Path(m).parts[0] for m in members if Path(m).parts}
    model_folder_name = zip_path.stem
    if len(root_candidates) == 1:
        model_folder_name = next(iter(root_candidates))
    return target_root / model_folder_name


def _model_has_required_files(model_dir: Path) -> bool:
    mesh_dir = model_dir / "Meshes"
    if not mesh_dir.is_dir():
        return False
    vtu_files = list(mesh_dir.glob("*.vtu"))
    if not vtu_files:
        return False
    paths_dir = model_dir / "Paths"
    if not paths_dir.is_dir() or not any(paths_dir.glob("*.pth")):
        return False
    for vtu_file in vtu_files:
        try:
            with vtu_file.open("rb") as handle:
                head = handle.read(128)
        except OSError:
            continue
        if b"<VTKFile" in head:
            return True
    return False


def ensure_model_available(model: str, vmr_pool: Path) -> Path:
    """Make sure ``model`` is present inside ``stEVEN/.data/vmr``.

    The function looks for either ``{model}.zip`` or an extracted directory
    inside ``vmr_pool``. Zip archives are extracted, while directories are
    copied. If neither is present, we fall back to the standard downloader.
    """

    internal_vmr_dir = _ensure_internal_vmr_dir()
    internal_model_dir = internal_vmr_dir / model
    if internal_model_dir.exists() and _model_has_required_files(internal_model_dir):
        return internal_model_dir

    vmr_pool = vmr_pool.expanduser().resolve()
    zip_candidate = vmr_pool / f"{model}.zip"
    dir_candidate = vmr_pool / model

    if zip_candidate.is_file():
        if internal_model_dir.exists():
            shutil.rmtree(internal_model_dir)
        extracted_root = _extract_zip(zip_candidate, internal_vmr_dir)
        if extracted_root != internal_model_dir and extracted_root.exists():
            if internal_model_dir.exists():
                shutil.rmtree(internal_model_dir)
            extracted_root.rename(internal_model_dir)
        LOGGER.info("Extracted %s to %s", zip_candidate, internal_model_dir)
        if _model_has_required_files(internal_model_dir):
            return internal_model_dir

    if dir_candidate.is_dir():
        if internal_model_dir.exists():
            shutil.rmtree(internal_model_dir)
        _copy_directory(dir_candidate, internal_model_dir)
        LOGGER.info("Copied %s to %s", dir_candidate, internal_model_dir)
        if _model_has_required_files(internal_model_dir):
            return internal_model_dir

    from eve.intervention.vesseltree.util.vmrdownload import download_vmr_files

    LOGGER.info(
        "Model %s not found under %s. Falling back to vascularmodel.com download.",
        model,
        vmr_pool,
    )
    downloaded_path = Path(download_vmr_files(model))
    if _model_has_required_files(downloaded_path):
        return downloaded_path

    raise RuntimeError(
        f"VMR model '{model}' is not usable after download. "
        "Ensure the remote dataset is accessible or provide a local copy via --vmr-pool."
    )


def list_available_models(vmr_pool: Path) -> list[str]:
    """Return model identifiers found in ``vmr_pool``."""

    vmr_pool = vmr_pool.expanduser().resolve()
    models: set[str] = set()
    if not vmr_pool.is_dir():
        return []
    for entry in vmr_pool.iterdir():
        if entry.name.startswith("."):
            continue
        if entry.is_file() and entry.suffix.lower() == ".zip":
            models.add(entry.stem)
        elif entry.is_dir():
            models.add(entry.name)
    return sorted(models)


def list_branch_names(model_dir: Path) -> list[str]:
    """Return available branch names for ``model_dir``."""

    path_dir = model_dir / "Paths"
    if not path_dir.exists():
        return []
    return sorted(p.stem.lower() for p in path_dir.glob("*.pth"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="VMR model identifier. If omitted, a random model from --vmr-pool is used.",
    )
    parser.add_argument(
        "--vmr-pool",
        type=Path,
        default=_default_vmr_pool(),
        help="Directory containing local VMR archives (default: data/VMR)",
    )
    parser.add_argument(
        "--insert-vessel",
        type=str,
        default=None,
        help="Branch name used for insertion (default: first available branch)",
    )
    parser.add_argument(
        "--insert-idx",
        type=int,
        default=1,
        help="Index of insertion point on the chosen branch",
    )
    parser.add_argument(
        "--direction-delta",
        type=int,
        default=2,
        help="Offset to determine insertion direction (along centerline)",
    )
    parser.add_argument(
        "--branch-radius",
        type=float,
        default=5.0,
        help="Approximate branch radius used for branching heuristics",
    )
    parser.add_argument(
        "--rotate",
        type=float,
        nargs=3,
        metavar=("Y", "Z", "X"),
        default=None,
        help="Optional YZX rotation (in degrees) applied to the vessel mesh",
    )
    parser.add_argument(
        "--no-mesh-check",
        action="store_true",
        help="Disable filtering centerline points outside the volumetric mesh",
    )
    parser.add_argument(
        "--branches",
        type=str,
        nargs="*",
        default=None,
        help="Target branches for CenterlineRandom (default: all available)",
    )
    parser.add_argument(
        "--list-branches",
        action="store_true",
        help="List branches for the resolved model and exit",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Disable visualisation (for benchmarking or headless operation)",
    )
    parser.add_argument(
        "--beam-scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor applied to the procedural device's beam counts. "
            "Values <1 reduce stiffness, values >1 increase it."
        ),
    )
    parser.add_argument(
        "--sim-dt",
        type=float,
        default=0.02,
        help="Simulation time step for SofaBeamAdapter (seconds per animate step).",
    )
    parser.add_argument(
        "--image-frequency",
        type=float,
        default=5.0,
        help="Fluoro/image frequency in Hz; lower values reduce per-frame Sofa steps.",
    )
    parser.add_argument(
        "--debug-insertion",
        action="store_true",
        help=(
            "Print a warning when the simulated wire base deviates from the "
            "configured insertion point."
        ),
    )
    parser.add_argument(
        "--insertion-tolerance",
        type=float,
        default=5.0,
        help=(
            "Maximum allowed distance (in model units) between the configured "
            "insertion point and the closest simulated node before warnings."
        ),
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=None,
        help="Optional trained policy checkpoint (.zip for SB3, .pt for diffusion) used for inference.",
    )
    parser.add_argument(
        "--policy-type",
        choices=["sb3", "diffusion"],
        default=None,
        help="Policy loader to use; defaults to SB3 for .zip, diffusion otherwise.",
    )
    parser.add_argument(
        "--policy-algo",
        choices=["ppo", "sac"],
        default="ppo",
        help="SB3 algorithm used to train the checkpoint.",
    )
    parser.add_argument(
        "--policy-deterministic",
        action="store_true",
        help="Use deterministic inference for SB3 policies.",
    )
    parser.add_argument(
        "--policy-device",
        type=str,
        default="auto",
        help="Device override for loaded policies (auto/cpu/cuda).",
    )
    return parser.parse_args()


def _build_policy_runner(env: eve.Env, args: argparse.Namespace):
    if args.policy_path is None:
        return None

    flatten_obs = functools.partial(gym.spaces.flatten, env.observation_space)
    flat_space = gym.spaces.flatten_space(env.observation_space)
    action_low = env.action_space.low
    action_high = env.action_space.high

    policy_type = args.policy_type
    if policy_type is None:
        policy_type = "sb3" if args.policy_path.suffix.lower() == ".zip" else "diffusion"

    if policy_type == "sb3":
        try:
            from stable_baselines3 import PPO, SAC
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("stable-baselines3 not installed; cannot load SB3 policy.") from exc
        ModelCls = PPO if args.policy_algo == "ppo" else SAC
        model = ModelCls.load(args.policy_path, device=args.policy_device)

        def _run(obs: dict) -> np.ndarray:
            flat = flatten_obs(obs)
            action, _ = model.predict(flat, deterministic=args.policy_deterministic)
            return np.asarray(action, dtype=np.float32)

        print(f"[autoendo] Loaded SB3 policy from {args.policy_path} ({args.policy_algo}).")
        return _run

    if policy_type == "diffusion":
        from .policy_diffusion import DiffusionPolicy

        ckpt = torch.load(
            args.policy_path,
            map_location=args.policy_device if args.policy_device != "auto" else "cpu",
        )
        obs_dim = int(ckpt.get("obs_dim", flat_space.shape[0]))
        action_dim = int(ckpt.get("action_dim", env.action_space.shape[0]))
        timesteps = int(ckpt.get("timesteps", 16))
        device = torch.device(
            args.policy_device if args.policy_device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        policy = DiffusionPolicy(obs_dim=obs_dim, action_dim=action_dim, timesteps=timesteps)
        policy.load_state_dict(ckpt["state_dict"])
        policy.to(device)
        policy.eval()

        def _run(obs: dict) -> np.ndarray:
            flat = flatten_obs(obs)
            obs_t = torch.as_tensor(flat, device=device, dtype=torch.float32).unsqueeze(0)
            action = policy.sample(obs_t).squeeze(0).cpu().numpy()
            action = np.clip(action, action_low, action_high)
            return action.astype(np.float32)

        print(f"[autoendo] Loaded diffusion policy from {args.policy_path} (timesteps={timesteps}).")
        return _run

    raise ValueError(f"Unknown policy_type {policy_type}")


def main() -> None:
    args = parse_args()
    available_models = list_available_models(args.vmr_pool)
    if not available_models:
        raise RuntimeError(
            f"No VMR models discovered under {args.vmr_pool}. "
            "Populate repo/data/vmr or pass --vmr-pool."
        )
    chosen_model = args.model or random.choice(available_models)
    if args.model is None:
        print(f"[autoendo] Using random VMR model: {chosen_model}")
    model_dir = ensure_model_available(chosen_model, args.vmr_pool)

    branch_names = list_branch_names(model_dir)
    if not branch_names:
        raise RuntimeError(f"No branches found in {model_dir}")

    if args.list_branches:
        print("Available branches:")
        for branch in branch_names:
            print(f" - {branch}")
        return

    insertion_vessel = (args.insert_vessel or random.choice(branch_names)).lower()
    if args.insert_vessel is None:
        print(f"[autoendo] Random insertion branch: {insertion_vessel}")
    if insertion_vessel not in branch_names:
        raise ValueError(
            f"Insertion vessel '{insertion_vessel}' not present. Candidates: {branch_names}"
        )

    target_branches = args.branches or [b for b in branch_names if b != insertion_vessel]
    if not target_branches:
        target_branches = branch_names

    vessel_tree = eve.intervention.vesseltree.VMR(
        model=chosen_model,
        insertion_vessel_name=insertion_vessel,
        insertion_point_idx=args.insert_idx,
        insertion_direction_idx_diff=args.direction_delta,
        approx_branch_radii=args.branch_radius,
        rotate_yzx_deg=tuple(args.rotate) if args.rotate else None,
        check_if_points_in_mesh=not args.no_mesh_check,
    )
    
    # Improved JShaped device instantiation with parameter overrides from args if present
    # # Allow for future extensibility: override device parameters via args if needed
    device_kwargs = {}
    if hasattr(args, "device_length"):
        device_kwargs["length"] = args.device_length
    if hasattr(args, "tip_angle"):
        device_kwargs["tip_angle"] = args.tip_angle
    if hasattr(args, "tip_radius"):
        device_kwargs["tip_radius"] = args.tip_radius

    device = eve.intervention.device.JShaped(**device_kwargs)

    # # Apply beam scaling if requested
    if args.beam_scale != 1.0:
        sofa_device = device.sofa_device
        # Scale density_of_beams tuple
        density = np.asarray(sofa_device.density_of_beams, dtype=float)
        scaled = np.maximum(1, np.round(density * args.beam_scale).astype(int))
        sofa_device.density_of_beams = tuple(int(v) for v in scaled)
        # Scale num_edges_collis tuple if present
        if sofa_device.num_edges_collis is not None:
            collis = np.asarray(sofa_device.num_edges_collis, dtype=float)
            sofa_device.num_edges_collis = tuple(
                int(max(1, round(c * args.beam_scale))) for c in collis
            )
    simulation = eve.intervention.simulation.SofaBeamAdapter(
        friction=0.1, dt_simulation=args.sim_dt
    )
    print("beam_density", device.sofa_device.density_of_beams)
    print("collis_edges", device.sofa_device.num_edges_collis)
    # print("visu_edges_per_mm", device.sofa_device.visu_edges_per_mm)
    # print("friction", simulation.friction)

    fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
        simulation=simulation,
        vessel_tree=vessel_tree,
        image_frequency=args.image_frequency,
        image_rot_zx=[20, 5],
    )

    target = eve.intervention.target.CenterlineRandom(
        vessel_tree=vessel_tree,
        fluoroscopy=fluoroscopy,
        threshold=5,
        branches=target_branches,
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

    position = eve.observation.Tracking2D(intervention=intervention, n_points=5)
    position = eve.observation.wrapper.NormalizeTracking2DEpisode(position, intervention)
    target_state = eve.observation.Target2D(intervention=intervention)
    target_state = eve.observation.wrapper.NormalizeTracking2DEpisode(
        target_state, intervention
    )
    rotation = eve.observation.Rotations(intervention=intervention)

    state = eve.observation.ObsDict(
        {"position": position, "target": target_state, "rotation": rotation}
    )

    target_reward = eve.reward.TargetReached(
        intervention=intervention,
        factor=1.0,
    )
    path_delta = eve.reward.PathLengthDelta(
        pathfinder=pathfinder,
        factor=0.01,
    )
    reward = eve.reward.Combination([target_reward, path_delta])

    target_reached = eve.terminal.TargetReached(intervention=intervention)
    max_steps = eve.truncation.MaxSteps(200)

    visualisation = SofaPygame(intervention=intervention)

    env = eve.Env(
        intervention=intervention,
        observation=state,
        reward=reward,
        terminal=target_reached,
        truncation=max_steps,
        visualisation=visualisation,
        start=start,
        pathfinder=pathfinder,
    )

    policy_runner = _build_policy_runner(env, args)

    insertion_warning_active = False
    obs, _info = env.reset()

    def check_insertion_alignment(source: str) -> None:
        """Check insertion alignment and log warnings if needed."""
        nonlocal insertion_warning_active
        if not args.debug_insertion:
            return
        measurement = simulation.measure_insertion_offset()
        if measurement is None:
            return
        distance, closest_point, expected_point = measurement
        if distance > args.insertion_tolerance:
            if not insertion_warning_active:
                expected_fmt = np.round(expected_point, 2).tolist()
                closest_fmt = np.round(closest_point, 2).tolist()
                msg = (
                    f"Insertion mismatch detected during {source}: distance={distance:.2f} "
                    f"(tolerance={args.insertion_tolerance}) expected={expected_fmt} "
                    f"closest={closest_fmt}"
                )
                LOGGER.warning(msg)
                print(msg)
            insertion_warning_active = True
        else:
            insertion_warning_active = False

    print("Environment initialized. Use arrow keys to advance the simulation.")
    check_insertion_alignment("reset")
    print("Press ENTER to reset, ESC to exit.")
    n_steps = 0
    while True:
        start_time = perf_counter()
        trans = 0.0
        rot = 0.0
        camera_trans = np.array([0.0, 0.0, 0.0])
        keys_pressed = None
        if not args.no_vis:
            pygame.event.get()
            keys_pressed = pygame.key.get_pressed()
            if keys_pressed[pygame.K_ESCAPE]:
                break
        if policy_runner is not None:
            action = policy_runner(obs)
        elif args.no_vis:
            trans = np.random.uniform(-5, 5)
            rot = np.random.choice([0, math.pi / 2, math.pi, 3 * math.pi / 2])
            action = (trans, rot)
        else:
            if keys_pressed[pygame.K_UP]:
                trans += 5
            if keys_pressed[pygame.K_DOWN]:
                trans -= 5
            if keys_pressed[pygame.K_LEFT]:
                rot += np.pi / 2
            if keys_pressed[pygame.K_RIGHT]:
                rot -= np.pi / 2
            if keys_pressed[pygame.K_r]:
                lao_rao = 0
                cra_cau = 0
                if keys_pressed[pygame.K_d]:
                    lao_rao += 10
                if keys_pressed[pygame.K_a]:
                    lao_rao -= 10
                if keys_pressed[pygame.K_w]:
                    cra_cau -= 10
                if keys_pressed[pygame.K_s]:
                    cra_cau += 10
                env.visualisation.rotate(lao_rao, cra_cau)
            else:
                if keys_pressed[pygame.K_w]:
                    camera_trans += np.array([0.0, 0.0, 200.0])
                if keys_pressed[pygame.K_s]:
                    camera_trans -= np.array([0.0, 0.0, 200.0])
                if keys_pressed[pygame.K_a]:
                    camera_trans -= np.array([200.0, 0.0, 0.0])
                if keys_pressed[pygame.K_d]:
                    camera_trans += np.array([200.0, 0.0, 0.0])
                env.visualisation.translate(camera_trans)
            if keys_pressed[pygame.K_e]:
                env.visualisation.zoom(1000)
            if keys_pressed[pygame.K_q]:
                env.visualisation.zoom(-1000)
            action = (trans, rot)
        
        t0 = perf_counter()
        obs, reward_value, terminal, truncation, info = env.step(action=action)

        check_insertion_alignment("step")

        t1 = perf_counter()
        env.render()

        n_steps += 1
        print({"steps": n_steps, "reward": reward_value, "terminal": terminal, "info": info})
        
        if policy_runner is not None and (terminal or truncation):
            obs, _ = env.reset()
            n_steps = 0
            insertion_warning_active = False
            check_insertion_alignment("policy-reset")
        elif not args.no_vis and keys_pressed is not None and keys_pressed[pygame.K_RETURN]:
            obs, _ = env.reset()
            n_steps = 0
            insertion_warning_active = False
            check_insertion_alignment("manual-reset")

        # Diagnostic FPS logging can help when tuning controllers.
        _ = perf_counter() - start_time

    env.close()

if __name__ == "__main__":
    main()
