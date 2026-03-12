"""
stroke.py — Stroke data model and preprocessing
Paper: Efficient Spatial Clustering and Recognition of Digital Ink
       in Infinite Canvas Interfaces
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


Point = Tuple[float, float]


@dataclass
class Stroke:
    """
    Represents a single pen-down → pen-up gesture on the canvas.
    points   : raw (x, y) coordinates captured during the stroke
    timestamp: when the stroke started (epoch ms) — used for temporal weighting
    stroke_id: unique identifier
    """
    points: List[Point]
    timestamp: float
    stroke_id: int
    label: int = -1          # ground-truth cluster label (-1 = unknown)

    # ------------------------------------------------------------------ #
    #  Geometric properties                                                #
    # ------------------------------------------------------------------ #

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Axis-aligned bounding box: (xmin, ymin, xmax, ymax)"""
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def center(self) -> Point:
        """Geometric center of the bounding box"""
        xmin, ymin, xmax, ymax = self.bbox
        return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)

    @property
    def width(self) -> float:
        xmin, _, xmax, _ = self.bbox
        return xmax - xmin

    @property
    def height(self) -> float:
        _, ymin, _, ymax = self.bbox
        return ymax - ymin

    @property
    def aspect_ratio(self) -> float:
        h = self.height
        return self.width / h if h > 0 else 0.0

    def __repr__(self):
        return (f"Stroke(id={self.stroke_id}, pts={len(self.points)}, "
                f"center=({self.center[0]:.1f},{self.center[1]:.1f}), "
                f"label={self.label})")


# ------------------------------------------------------------------ #
#  Preprocessing                                                       #
# ------------------------------------------------------------------ #

def resample_stroke(stroke: Stroke, n_points: int = 32) -> Stroke:
    """
    Resample a stroke to exactly n_points using linear interpolation.
    This normalises stroke density — fast strokes have fewer raw points
    than slow ones, which would otherwise bias distance metrics.
    """
    pts = np.array(stroke.points, dtype=float)
    if len(pts) < 2:
        # Pad with the single point repeated
        pts = np.tile(pts, (n_points, 1))
        return Stroke(pts.tolist(), stroke.timestamp, stroke.stroke_id, stroke.label)

    # Compute cumulative arc-length
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_len = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_len = cum_len[-1]

    if total_len == 0:
        pts = np.tile(pts[0], (n_points, 1))
        return Stroke(pts.tolist(), stroke.timestamp, stroke.stroke_id, stroke.label)

    # Evenly-spaced sample positions along arc length
    sample_pos = np.linspace(0, total_len, n_points)
    new_x = np.interp(sample_pos, cum_len, pts[:, 0])
    new_y = np.interp(sample_pos, cum_len, pts[:, 1])

    resampled = list(zip(new_x.tolist(), new_y.tolist()))
    return Stroke(resampled, stroke.timestamp, stroke.stroke_id, stroke.label)


def smooth_stroke(stroke: Stroke, window: int = 3) -> Stroke:
    """
    Apply a simple moving-average filter to reduce digitiser noise.
    window=3 is gentle enough to preserve character shape.
    """
    pts = np.array(stroke.points, dtype=float)
    if len(pts) <= window:
        return stroke

    kernel = np.ones(window) / window
    smooth_x = np.convolve(pts[:, 0], kernel, mode='same')
    smooth_y = np.convolve(pts[:, 1], kernel, mode='same')

    smoothed = list(zip(smooth_x.tolist(), smooth_y.tolist()))
    return Stroke(smoothed, stroke.timestamp, stroke.stroke_id, stroke.label)


def normalize_stroke(stroke: Stroke) -> Stroke:
    """
    Translate stroke so its bounding box starts at (0, 0).
    Useful before rendering clusters to a fixed-size image.
    """
    xmin, ymin, _, _ = stroke.bbox
    shifted = [(x - xmin, y - ymin) for x, y in stroke.points]
    return Stroke(shifted, stroke.timestamp, stroke.stroke_id, stroke.label)


def preprocess(stroke: Stroke,
               n_points: int = 32,
               smooth_window: int = 3) -> Stroke:
    """Full preprocessing chain: resample → smooth"""
    s = resample_stroke(stroke, n_points)
    s = smooth_stroke(s, smooth_window)
    return s