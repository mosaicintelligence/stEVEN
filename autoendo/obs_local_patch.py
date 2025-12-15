"""
obs_local_patch.py

Local centerline patch observation shared between training and autonomy_test.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class LocalCenterlinePatch:
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
