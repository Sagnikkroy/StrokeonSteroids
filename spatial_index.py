"""
spatial_index.py — KD-tree indexing for fast neighbour search
Paper: Efficient Spatial Clustering and Recognition of Digital Ink
       in Infinite Canvas Interfaces

WHY THIS MATTERS:
  Naive pairwise distance check  →  O(n²)
  KD-tree query_ball_tree        →  O(n log n)
  At n = 10,000 strokes that's 10,000 ops vs ~130 ops per query.
"""

from typing import List, Dict
import numpy as np
from scipy.spatial import KDTree

from stroke import Stroke


class StrokeSpatialIndex:
    """
    Wraps a KDTree built from stroke bounding-box centers.
    Supports radius-based neighbour queries in O(log n).
    """

    def __init__(self, strokes: List[Stroke]):
        if not strokes:
            raise ValueError("Cannot build index from empty stroke list.")

        self.strokes = strokes
        self._centers = np.array([s.center for s in strokes], dtype=float)
        self._tree = KDTree(self._centers)

        print(f"[SpatialIndex] Built KD-tree over {len(strokes)} strokes.")

    def query_radius(self, stroke_idx: int, radius: float) -> List[int]:
        """
        Return indices of all strokes whose center falls within
        `radius` pixels of strokes[stroke_idx].
        """
        return self._tree.query_ball_point(
            self._centers[stroke_idx], r=radius
        )

    def query_point(self, x: float, y: float, radius: float) -> List[int]:
        """Find all strokes near an arbitrary canvas coordinate."""
        return self._tree.query_ball_point([x, y], r=radius)

    def all_neighbour_pairs(self, radius: float) -> List[List[int]]:
        """
        For every stroke, return its neighbour list.
        This is the input DBSCAN needs.
        Returns a list of length n, each element is a list of neighbour indices.
        """
        return self._tree.query_ball_tree(self._tree, r=radius)

    @property
    def centers(self) -> np.ndarray:
        return self._centers