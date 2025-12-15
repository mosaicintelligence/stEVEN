# pylint: disable=unused-argument
from typing import Any, Dict, List, Optional
import gymnasium as gym

import numpy as np
from .intervention import SimulatedIntervention
from .target import Target
from .vesseltree import VesselTree
from .vesseltree.vesseltree import Insertion, at_tree_end, find_nearest_branch_to_point
from .fluoroscopy import SimulatedFluoroscopy
from .device import Device
from .simulation import Simulation


class MonoPlaneStatic(SimulatedIntervention):
    def __init__(
        self,
        vessel_tree: VesselTree,
        devices: List[Device],
        simulation: Simulation,
        fluoroscopy: SimulatedFluoroscopy,
        target: Target,
        stop_device_at_tree_end: bool = True,
        normalize_action: bool = False,
    ) -> None:
        self.vessel_tree = vessel_tree
        self.devices = devices
        self.target = target
        self.fluoroscopy = fluoroscopy
        self.stop_device_at_tree_end = stop_device_at_tree_end
        self.normalize_action = normalize_action
        self.simulation = simulation
        self._np_random = np.random.default_rng()

        self.velocity_limits = np.array(
            [device.velocity_limit for device in self.devices], dtype=np.float32
        )
        # Replace any non-finite entries with a sane default to keep action_space bounded.
        if not np.all(np.isfinite(self.velocity_limits)):
            self.velocity_limits = np.nan_to_num(
                self.velocity_limits, nan=0.0, posinf=1e3, neginf=1e3
            )
        self.velocity_limits = np.clip(self.velocity_limits, -1e6, 1e6)
        limits = np.abs(self.velocity_limits).reshape(-1)
        print(f"[Intervention] velocity limits: {limits}")
        self._action_space = gym.spaces.Box(low=-limits, high=limits, dtype=np.float32)
        print(f"[Intervention] action_space low={self._action_space.low}, high={self._action_space.high}")
        self.last_action = np.zeros_like(self.velocity_limits)
        self._device_lengths_inserted = self.simulation.inserted_lengths
        self._device_rotations = self.simulation.rotations

    @property
    def device_lengths_inserted(self) -> List[float]:
        return self.simulation.inserted_lengths

    @property
    def device_rotations(self) -> List[float]:
        return self.simulation.rotations

    @property
    def device_lengths_maximum(self) -> List[float]:
        return [device.length for device in self.devices]

    @property
    def device_diameters(self) -> List[float]:
        return [device.sofa_device.radius * 2 for device in self.devices]

    @property
    def action_space(self) -> gym.spaces.Box:
        if self.normalize_action:
            high = np.ones_like(self.velocity_limits, dtype=np.float32)
            return gym.spaces.Box(low=-high, high=high, dtype=np.float32)
        return self._action_space

    def step(self, action: np.ndarray) -> None:
        action = np.array(action).reshape(self.velocity_limits.shape)
        if self.normalize_action:
            action = np.clip(action, -1.0, 1.0)
            self.last_action = action
            high = self.velocity_limits
            low = -high
            action = (action + 1) / 2 * (high - low) + low
        else:
            action = np.clip(action, -self.velocity_limits, self.velocity_limits)
            self.last_action = action

        inserted_lengths = np.array(self.device_lengths_inserted)
        max_lengths = np.array(self.device_lengths_maximum)
        duration = 1 / self.fluoroscopy.image_frequency
        mask = np.where(inserted_lengths + action[:, 0] * duration <= 0.0)
        action[mask, 0] = 0.0
        mask = np.where(inserted_lengths + action[:, 0] * duration >= max_lengths)
        action[mask, 0] = 0.0
        tip = self.simulation.dof_positions[0]
        if self.stop_device_at_tree_end and at_tree_end(tip, self.vessel_tree):
            max_length = max(inserted_lengths)
            if max_length > 10:
                dist_to_longest = -1 * inserted_lengths + max_length
                movement = action[:, 0] * duration
                mask = movement > dist_to_longest
                action[mask, 0] = 0.0

        self.vessel_tree.step()
        self.simulation.step(action, duration)
        self.fluoroscopy.step()
        self.target.step()

    def reset(
        self,
        episode_number: int = 0,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        vessel_seed = None if seed is None else self._np_random.integers(0, 2**31)
        self.vessel_tree.reset(episode_number, vessel_seed)
        ip_pos = self.vessel_tree.insertion.position
        ip_dir = self.vessel_tree.insertion.direction
        self.simulation.reset(
            insertion_point=ip_pos,
            insertion_direction=ip_dir,
            mesh_path=self.vessel_tree.mesh_path,
            devices=self.devices,
            centerlines=[branch.coordinates for branch in self.vessel_tree.branches],
            coords_low=self.vessel_tree.coordinate_space.low,
            coords_high=self.vessel_tree.coordinate_space.high,
            vessel_visual_path=self.vessel_tree.visu_mesh_path,
        )
        target_seed = None if seed is None else self._np_random.integers(0, 2**31)
        self.target.reset(episode_number, target_seed)
        # After target is known, re-aim insertion direction along the first segment of the
        # shortest centerline path (instead of a direct straight-line vector).
        try:
            target_coord = np.array(self.target.coordinates3d, dtype=np.float32)
            path_dir = self._compute_centerline_path_dir(ip_pos, target_coord)
            if path_dir is None:
                # Fallback: straight toward target.
                vec = target_coord - ip_pos
                norm = np.linalg.norm(vec)
                path_dir = vec / norm if norm > 1e-6 else None
            if path_dir is not None:
                self.vessel_tree.insertion = Insertion(ip_pos, path_dir)
                self.simulation.reset(
                    insertion_point=ip_pos,
                    insertion_direction=path_dir,
                    mesh_path=self.vessel_tree.mesh_path,
                    devices=self.devices,
                    centerlines=[branch.coordinates for branch in self.vessel_tree.branches],
                    coords_low=self.vessel_tree.coordinate_space.low,
                    coords_high=self.vessel_tree.coordinate_space.high,
                    vessel_visual_path=self.vessel_tree.visu_mesh_path,
                )
                print(f"[Insertion] reoriented along centerline path dir={np.round(path_dir,3).tolist()}")
        except Exception as exc:  # pragma: no cover
            print(f"[Insertion] failed to reorient toward path: {exc}")
        self.fluoroscopy.reset(episode_number)
        self.last_action *= 0.0

    def _compute_centerline_path_dir(self, start: np.ndarray, target: np.ndarray) -> Optional[np.ndarray]:
        """Compute the direction of the first segment along the shortest centerline path to target."""
        try:
            bps = getattr(self.vessel_tree, "branching_points", None)
            if bps is None or len(bps) == 0 or self.vessel_tree.branches is None:
                return None
            start_branch = find_nearest_branch_to_point(start, self.vessel_tree)
            target_branch = find_nearest_branch_to_point(target, self.vessel_tree)
            if start_branch is None or target_branch is None:
                return None

            def get_length(path: np.ndarray) -> float:
                return float(np.sum(np.linalg.norm(path[:-1] - path[1:], axis=1)))

            # Build node connections between branching points.
            node_connections = {}
            for bp in bps:
                node_connections[bp] = {}
                for connection in bp.connections:
                    for other_bp in bps:
                        if bp == other_bp:
                            continue
                        if connection in other_bp.connections:
                            pts = connection.get_path_along_branch(bp.coordinates, other_bp.coordinates)
                            node_connections[bp][other_bp] = (get_length(pts), pts)

            # Base graph adjacency.
            base_graph = {bp: list(node_connections.get(bp, {}).keys()) for bp in node_connections}

            def create_search_graph():
                graph = {k: list(v) for k, v in base_graph.items()}
                if start_branch == target_branch:
                    graph["start"] = ["target"]
                    return graph
                start_conns = []
                for bp in bps:
                    if start_branch in bp.connections:
                        start_conns.append(bp)
                    if target_branch in bp.connections:
                        graph[bp].append("target")
                graph["start"] = start_conns
                return graph

            def bfs_paths(graph):
                queue = [("start", ["start"])]
                while queue:
                    vertex, path = queue.pop(0)
                    for nxt in graph.get(vertex, []):
                        if nxt in path:
                            continue
                        if nxt == "target":
                            yield path + [nxt]
                        else:
                            queue.append((nxt, path + [nxt]))

            def shortest_path_points():
                graph = create_search_graph()
                paths = bfs_paths(graph)
                best_len = np.inf
                best_pts = None
                first_path = next(paths, None)
                if first_path is None:
                    return None
                # restart iteration including first_path
                for path in [first_path] + list(paths):
                    if len(path) == 2:  # same branch
                        pts = start_branch.get_path_along_branch(start, target)
                        path_len = get_length(pts)
                    else:
                        pts = start_branch.get_path_along_branch(start, path[1].coordinates)
                        path_len = get_length(pts)
                        for node, nxt in zip(path[1:-2], path[2:-1]):
                            length, seg = node_connections[node][nxt]
                            path_len += length
                            pts = np.vstack((pts, seg[1:]))
                        tail = target_branch.get_path_along_branch(path[-2].coordinates, target)
                        path_len += get_length(tail)
                        pts = np.vstack((pts, tail[1:]))
                    if path_len < best_len:
                        best_len = path_len
                        best_pts = pts
                return best_pts

            pts = shortest_path_points()
            if pts is None or pts.shape[0] < 2:
                return None
            direction = pts[1] - pts[0]
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                return None
            return direction / norm
        except Exception as exc:  # pragma: no cover
            print(f"[Insertion] path dir compute failed: {exc}")
            return None

    def _update_states(self):
        self._device_lengths_inserted = self.simulation.inserted_lengths
        self._device_rotations = self.simulation.rotations

    def close(self) -> None:
        self.simulation.close()

    def reset_devices(self) -> None:
        self.simulation.reset_devices()
