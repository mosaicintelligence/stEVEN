"""Interactive single J-wire test scene for VMR vessel models.

This script mirrors ``single_jwire.py`` but allows loading arbitrary VMR
datasets that are locally available in ``data/VMR``. The chosen model is copied
or extracted into ``stEVEN/.data/vmr`` so it can be consumed by the standard
``eve.intervention.vesseltree.VMR`` loader. Run ``python vmr_single_jwire.py --help``
for command-line usage details.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import zipfile
from time import perf_counter

import numpy as np
import pygame

import eve
from eve.visualisation.sofapygame import SofaPygame


LOGGER = logging.getLogger(__name__)


def _default_vmr_pool() -> Path:
    """Return the repo-local ``data/VMR`` directory."""

    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "VMR"


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


def list_branch_names(model_dir: Path) -> list[str]:
    """Return available branch names for ``model_dir``."""

    path_dir = model_dir / "Paths"
    if not path_dir.exists():
        return []
    return sorted(p.stem.lower() for p in path_dir.glob("*.pth"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="VMR model identifier, e.g. 0049_H_ABAO_AIOD",
    )
    parser.add_argument(
        "--model",
        dest="model_override",
        default=None,
        help="VMR model identifier, mirrors the positional argument",
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
        default=25,
        help="Index of insertion point on the chosen branch",
    )
    parser.add_argument(
        "--direction-delta",
        type=int,
        default=1,
        help="Offset between insertion indices to determine insertion direction",
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
        "--beam-scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor applied to the procedural device's beam counts. "
            "Values <1 reduce stiffness, values >1 increase it."
        ),
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = args.model_override or args.model or "0049_H_ABAO_AIOD"
    model_dir = ensure_model_available(model, args.vmr_pool)

    branch_names = list_branch_names(model_dir)
    if not branch_names:
        raise RuntimeError(f"No branches found in {model_dir}")

    if args.list_branches:
        print("Available branches:")
        for branch in branch_names:
            print(f" - {branch}")
        return

    insertion_vessel = args.insert_vessel or branch_names[0]
    if insertion_vessel not in branch_names:
        raise ValueError(
            f"Insertion vessel '{insertion_vessel}' not present. Candidates: {branch_names}"
        )

    target_branches = args.branches or [b for b in branch_names if b != insertion_vessel]
    if not target_branches:
        target_branches = branch_names

    vessel_tree = eve.intervention.vesseltree.VMR(
        model=model,
        insertion_vessel_name=insertion_vessel,
        insertion_point_idx=args.insert_idx,
        insertion_direction_idx_diff=args.direction_delta,
        approx_branch_radii=args.branch_radius,
        rotate_yzx_deg=tuple(args.rotate) if args.rotate else None,
        check_if_points_in_mesh=not args.no_mesh_check,
    )

    device = eve.intervention.device.JShaped()
    if args.beam_scale != 1.0:
        sofa_device = device.sofa_device
        density = np.asarray(sofa_device.density_of_beams, dtype=float)
        scaled = np.maximum(1, np.round(density * args.beam_scale).astype(int))
        sofa_device.density_of_beams = tuple(int(v) for v in scaled)
        if sofa_device.num_edges_collis is not None:
            collis = np.asarray(sofa_device.num_edges_collis, dtype=float)
            sofa_device.num_edges_collis = tuple(
                int(max(1, round(c * args.beam_scale))) for c in collis
            )
    simulation = eve.intervention.simulation.SofaBeamAdapter(friction=0.001)

    fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
        simulation=simulation,
        vessel_tree=vessel_tree,
        image_frequency=7.5,
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

    insertion_warning_active = False

    def check_insertion_alignment(source: str) -> None:
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

    env.reset()
    check_insertion_alignment("reset")

    n_steps = 0
    while True:
        start_time = perf_counter()
        trans = 0.0
        rot = 0.0
        camera_trans = np.array([0.0, 0.0, 0.0])
        pygame.event.get()
        keys_pressed = pygame.key.get_pressed()

        if keys_pressed[pygame.K_ESCAPE]:
            break
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
        obs, reward_value, terminal, truncation, info = env.step(action=action)
        check_insertion_alignment("step")
        env.render()
        n_steps += 1
        print({"steps": n_steps, "reward": reward_value, "terminal": terminal, "info": info})
        if keys_pressed[pygame.K_RETURN]:
            env.reset()
            n_steps = 0
            insertion_warning_active = False
            check_insertion_alignment("manual-reset")

        # Diagnostic FPS logging can help when tuning controllers.
        _ = perf_counter() - start_time

    env.close()


if __name__ == "__main__":
    main()
