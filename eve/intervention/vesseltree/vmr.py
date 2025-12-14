from xml.dom import minidom
from typing import List, Optional, Tuple, Union
import os
import numpy as np
import pyvista as pv

from .vesseltree import VesselTree, Insertion, gym
from .util.branch import Branch, calc_branching, rotate_branches
from .util import calc_insertion, calc_insertion_from_branch_start
from .util.meshing import get_temp_mesh_path
from .util.vmrdownload import download_vmr_files


SCALING_FACTOR = 10
LOW_HIGH_BUFFER = 3


def _get_branches(model_dir, vtu_mesh, check_if_points_in_mesh: bool) -> List[Branch]:
    path_dir = os.path.join(model_dir, "Paths")
    files = _get_available_pths(path_dir)
    files = sorted(files)
    branches = []
    for file in files:
        file_path = os.path.join(path_dir, file)
        branch = _load_points_from_pth(file_path, vtu_mesh, check_if_points_in_mesh)
        if branch.coordinates.size > 3:
            branches.append(branch)
    return branches


def _get_available_pths(directory: str) -> List[str]:
    pth_files = []
    for file in os.listdir(directory):
        if file.endswith(".pth"):
            pth_files.append(file)
    return pth_files


def _get_vtk_file(directory: str, file_ending: str) -> str:
    for file in os.listdir(directory):
        if file.endswith(file_ending):
            path = os.path.join(directory, file)
            return path

def _load_raw_points_from_pth(pth_file_path: str) -> np.ndarray:
    """Load all centerline points from a .pth without filtering against the mesh."""
    with open(pth_file_path, "r", encoding="utf-8") as file:
        next(file)
        next(file)
        tree = minidom.parse(file)
    points = []
    for point in tree.getElementsByTagName("pos"):
        x = float(point.attributes["x"].value) * SCALING_FACTOR
        y = float(point.attributes["y"].value) * SCALING_FACTOR
        z = float(point.attributes["z"].value) * SCALING_FACTOR
        points.append([x, y, z])
    return np.array(points, dtype=np.float32)


def _load_points_from_pth(
    pth_file_path: str, vtu_mesh: pv.UnstructuredGrid, check_if_points_in_mesh: bool
) -> Branch:
    points = []
    name = os.path.basename(pth_file_path)[:-4]
    with open(pth_file_path, "r", encoding="utf-8") as file:
        next(file)
        next(file)
        tree = minidom.parse(file)

    xml_points = tree.getElementsByTagName("pos")

    points = []
    low = np.array([vtu_mesh.bounds[0], vtu_mesh.bounds[2], vtu_mesh.bounds[4]])
    high = np.array([vtu_mesh.bounds[1], vtu_mesh.bounds[3], vtu_mesh.bounds[5]])
    low += LOW_HIGH_BUFFER
    high -= LOW_HIGH_BUFFER
    for point in xml_points:
        x = float(point.attributes["x"].value) * SCALING_FACTOR
        y = float(point.attributes["y"].value) * SCALING_FACTOR
        z = float(point.attributes["z"].value) * SCALING_FACTOR
        if np.any([x, y, z] < low) or np.any([x, y, z] > high):
            continue
        points.append([x, y, z])
    points = np.array(points, dtype=np.float32)
    if check_if_points_in_mesh:
        to_keep = vtu_mesh.find_containing_cell(points) + 1
        to_keep = np.argwhere(to_keep)
        points = points[to_keep.reshape(-1)]
    return Branch(
        name=name.lower(),
        coordinates=np.array(points, dtype=np.float32),
    )


class VMR(VesselTree):
    def __init__(
        self,
        model: str,
        insertion_vessel_name: str,
        insertion_point_idx: int, # Index of insertion point on chosen branch
        insertion_direction_idx_diff: int,
        approx_branch_radii: Union[List[float], float],
        rotate_yzx_deg: Optional[Tuple[float, float, float]] = None,
        check_if_points_in_mesh: bool = True,
        auto_choose_insertion_endpoint: bool = True,
    ) -> None:
        self.model = model
        self.insertion_point_idx = insertion_point_idx
        self.insertion_direction_idx_diff = insertion_direction_idx_diff
        self.insertion_vessel_name = insertion_vessel_name.lower()
        self.approx_branch_radii = approx_branch_radii
        self.rotate_yzx_deg = rotate_yzx_deg
        self.check_if_points_in_mesh = check_if_points_in_mesh
        self.auto_choose_insertion_endpoint = auto_choose_insertion_endpoint

        self._model_folder = download_vmr_files(model)
        self.mesh_folder = os.path.join(self._model_folder, "Meshes")

        branches = self._read_branches()
        self.coordinate_space = self._calc_coord_space(branches)
        self.coordinate_space_episode = self.coordinate_space

        self.branches = None
        self.insertion = None
        self.branching_points = None
        self.centerline_coordinates = None

        self._mesh_path = None

    @property
    def mesh_path(self) -> str:
        if self._mesh_path is None:
            self._make_mesh_obj()
        return self._mesh_path

    @property
    def visu_mesh_path(self) -> str:
        return self.mesh_path

    def reset(self, episode_nr=0, seed: int = None) -> None:
        if self.branches is None:
            self._make_branches()

    def _make_branches(self):
        branches = self._read_branches()
        self.branches = branches

        self.coordinate_space = self._calc_coord_space(branches)
        centerline_coordinates = [branch.coordinates for branch in branches]
        self.centerline_coordinates = np.concatenate(centerline_coordinates)

        insert_vessel = self[self.insertion_vessel_name]
        raw_insert_branch = self._load_raw_insertion_branch(insert_vessel.name)
        ip_idx = self.insertion_point_idx
        dir_idx = self.insertion_point_idx + self.insertion_direction_idx_diff
        if self.auto_choose_insertion_endpoint:
            ip_idx, dir_idx = self._pick_insertion_endpoint(raw_insert_branch)
        ip_idx = int(np.clip(ip_idx, 0, raw_insert_branch.coordinates.shape[0] - 1))
        dir_idx = int(np.clip(dir_idx, 0, raw_insert_branch.coordinates.shape[0] - 1))
        ip_raw, ip_dir_raw = calc_insertion(raw_insert_branch, ip_idx, dir_idx)

        # Fallback: if the raw insertion is clearly outside the vessel bounding box,
        # reuse the filtered branch to keep the start near the visible geometry.
        fallback_ip, fallback_dir = calc_insertion(
            insert_vessel,
            np.clip(self.insertion_point_idx, 0, insert_vessel.coordinates.shape[0] - 1),
            np.clip(
                self.insertion_point_idx + self.insertion_direction_idx_diff,
                0,
                insert_vessel.coordinates.shape[0] - 1,
            ),
        )
        if (
            np.any(ip_raw < self.coordinate_space.low)
            or np.any(ip_raw > self.coordinate_space.high)
            or np.linalg.norm(ip_raw - fallback_ip) > 20.0
        ):
            ip, ip_dir = fallback_ip, fallback_dir
        else:
            ip, ip_dir = ip_raw, ip_dir_raw
        self.insertion = Insertion(ip, ip_dir)
        print(f"[VMR] insertion branch={insert_vessel.name} ip={np.round(ip,2)} dir={np.round(ip_dir,2)}")
        self.branching_points = calc_branching(self.branches, self.approx_branch_radii)
        self._mesh_path = None

    def _read_branches(self):
        mesh_path = _get_vtk_file(self.mesh_folder, ".vtu")
        mesh = pv.read(mesh_path)
        mesh.scale([SCALING_FACTOR, SCALING_FACTOR, SCALING_FACTOR], inplace=True)
        branches = _get_branches(self._model_folder, mesh, self.check_if_points_in_mesh)

        if self.rotate_yzx_deg is not None:
            branches = rotate_branches(branches, self.rotate_yzx_deg)
        return branches

    def _calc_coord_space(self, branches):
        branch_highs = [branch.high for branch in branches]
        high = np.max(branch_highs, axis=0)
        branch_lows = [branch.low for branch in branches]
        low = np.min(branch_lows, axis=0)
        return gym.spaces.Box(low=low, high=high)

    def _pick_insertion_endpoint(self, branch: Branch) -> Tuple[int, int]:
        """Choose the branch end closest to the bounding box as insertion point."""
        coords = branch.coordinates
        n_points = coords.shape[0]
        if n_points < 2:
            return 0, min(1, n_points - 1)

        low = self.coordinate_space.low
        high = self.coordinate_space.high

        def boundary_distance(pt: np.ndarray) -> float:
            return float(np.min(np.concatenate([pt - low, high - pt])))

        first_dist = boundary_distance(coords[0])
        last_dist = boundary_distance(coords[-1])

        if last_dist < first_dist:
            return n_points - 1, n_points - 2
        return 0, 1

    def _load_raw_insertion_branch(self, branch_name: str) -> Branch:
        """Load the chosen insertion branch without any mesh/bounds filtering."""
        path_dir = os.path.join(self._model_folder, "Paths")
        for file in os.listdir(path_dir):
            if file.endswith(".pth") and file[:-4].lower() == branch_name:
                raw_points = _load_raw_points_from_pth(os.path.join(path_dir, file))
                branch = Branch(branch_name, raw_points)
                if self.rotate_yzx_deg is not None:
                    # Apply the same rotation as the mesh/branches to keep insertion aligned
                    branch = rotate_branches([branch], self.rotate_yzx_deg)[0]
                return branch
        # Fallback to filtered branch if raw not found
        return self[branch_name]

    def _make_mesh_obj(self):
        mesh_path = _get_vtk_file(self.mesh_folder, ".vtp")
        mesh = pv.read(mesh_path)
        mesh.flip_normals()
        mesh.scale([SCALING_FACTOR, SCALING_FACTOR, SCALING_FACTOR], inplace=True)
        if self.rotate_yzx_deg is not None:
            mesh.rotate_y(self.rotate_yzx_deg[0], inplace=True)
            mesh.rotate_z(self.rotate_yzx_deg[1], inplace=True)
            mesh.rotate_x(self.rotate_yzx_deg[2], inplace=True)
        mesh.decimate(0.9, inplace=True)

        obj_mesh_path = get_temp_mesh_path("VMR")
        pv.save_meshio(obj_mesh_path, mesh)
        self._mesh_path = obj_mesh_path
