"""
adaptive_clustering.py — Scale-Invariant Adaptive ε Clustering
Paper: Efficient Spatial Clustering and Recognition of Digital Ink
       in Infinite Canvas Interfaces

THE PROBLEM WITH FIXED ε:
───────────────────────────────────────────────────────────────
  Imagine a canvas with:
    - Word A written large  → strokes 200px apart
    - Word B written small  → strokes 20px apart
    - Word C written tiny   → strokes 5px apart

  A fixed ε = 80px:
    → Merges Word B and Word C into one cluster  ✗
    → Fragments Word A into many clusters        ✗

THE FIX — Adaptive ε per stroke:
───────────────────────────────────────────────────────────────
  For each stroke i, compute ε_i based on the stroke's own
  bounding box size:

      ε_i = α × diag(bbox_i)

  Where diag = sqrt(width² + height²) of the stroke's bbox.
  α is a scale factor (default 2.5) — tunable.

  Intuition: a stroke that is 100px tall should cluster with
  neighbours up to ~250px away. A stroke that is 10px tall
  should only cluster with neighbours within ~25px.

  WHY THIS WORKS FOR HANDWRITING:
  - Strokes in the same character are always proportional in
    size to each other (you don't write one giant stroke and
    one tiny stroke in the same letter)
  - Inter-character spacing is also roughly proportional to
    character size (large handwriting → large gaps between chars)

ALGORITHM: Adaptive DBSCAN
───────────────────────────────────────────────────────────────
  Standard DBSCAN uses one global ε.
  We extend it: each point has its own ε_i.

  Two strokes i and j are neighbours if:
      dist(center_i, center_j) ≤ max(ε_i, ε_j)

  We use max() so the relationship is symmetric:
  if either stroke "reaches" the other, they connect.
  (You could also use min() for stricter grouping — we test both.)
"""

import numpy as np
from typing import List, Tuple, Dict
from scipy.spatial import KDTree
import time

from stroke import Stroke


ClusterMap = Dict[int, List[Stroke]]


# ──────────────────────────────────────────────
#  Local density based epsilon estimation
# ──────────────────────────────────────────────

def compute_adaptive_eps(
    strokes: List[Stroke],
    alpha: float = 2.5,
    k: int = 2
) -> np.ndarray:
    """
    For each stroke i, estimate ε_i from its local neighbourhood density:

        ε_i = alpha × mean_distance_to_k_nearest_neighbours

    WHY THIS WORKS:
    ───────────────
    The k-nearest-neighbour distance is a direct measure of local writing
    scale — it captures how tightly packed the strokes around stroke i are,
    regardless of absolute canvas size or character size.

    - Large handwriting  → nn distances ~150px → ε_i ~375px
    - Medium handwriting → nn distances ~ 50px → ε_i ~125px
    - Tiny handwriting   → nn distances ~ 12px → ε_i ~ 30px

    Each word self-calibrates. No global tuning needed.

    Parameters
    ----------
    strokes : list of Stroke objects
    alpha   : multiplier on local scale. Controls clustering tightness.
              Typical range: 1.5 – 3.0. Default 2.5 works well.
    k       : number of nearest neighbours to average over.
              k=2 is robust for sparse canvases.

    Returns
    -------
    eps_array : shape (n,) float array of per-stroke epsilon values
    """
    if len(strokes) <= k:
        centers = np.array([s.center for s in strokes])
        fallback = np.mean(np.std(centers, axis=0)) * alpha
        return np.full(len(strokes), max(fallback, 20.0))

    centers = np.array([s.center for s in strokes])
    tree = KDTree(centers)

    # k+1 because index 0 is always self (distance=0)
    dists, _ = tree.query(centers, k=k + 1)
    mean_nn_dist = dists[:, 1:].mean(axis=1)   # skip self at col 0

    eps_array = alpha * mean_nn_dist
    eps_array = np.maximum(eps_array, 10.0)    # floor at 10px
    return eps_array


# ──────────────────────────────────────────────
#  Adaptive DBSCAN
# ──────────────────────────────────────────────

def adaptive_dbscan(
    strokes: List[Stroke],
    alpha: float = 1.5,
    min_samples: int = 1,
    sym: str = 'min'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DBSCAN with per-stroke adaptive epsilon.

    Parameters
    ----------
    strokes     : list of preprocessed Stroke objects
    alpha       : scale multiplier for ε_i = alpha × diag(stroke_i)
    min_samples : minimum neighbourhood size to be a core point
    sym         : 'max' — use max(ε_i, ε_j) for symmetry (more inclusive)
                  'min' — use min(ε_i, ε_j)  (stricter)

    Returns
    -------
    labels      : cluster label per stroke (-1 = noise)
    eps_array   : the per-stroke epsilon values (useful for visualisation)
    """
    n = len(strokes)
    if n == 0:
        return np.array([]), np.array([])

    centers   = np.array([s.center for s in strokes])   # (n, 2)
    eps_array = compute_adaptive_eps(strokes, alpha, k=2)  # (n,)

    # Build KD-tree on centers — O(n log n)
    tree = KDTree(centers)

    # For each stroke, find candidates within max possible ε
    # We use max(eps_array) as a safe upper bound for the initial query,
    # then filter by per-pair rule. This keeps KD-tree benefit.
    global_max_eps = float(np.max(eps_array))

    # Get all candidate neighbour pairs within global_max_eps
    candidate_pairs = tree.query_ball_tree(tree, r=global_max_eps)  # O(n log n)

    # Filter each candidate list by the per-pair adaptive rule
    def neighbours(i: int) -> List[int]:
        result = []
        for j in candidate_pairs[i]:
            if i == j:
                continue
            dist = np.linalg.norm(centers[i] - centers[j])
            threshold = (max if sym == 'max' else min)(eps_array[i], eps_array[j])
            if dist <= threshold:
                result.append(j)
        return result

    # Standard DBSCAN expansion with adaptive neighbour function
    labels   = np.full(n, -1, dtype=int)
    visited  = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nb = neighbours(i)

        if len(nb) < min_samples:
            labels[i] = -1   # noise for now — may be absorbed later
            continue

        labels[i] = cluster_id
        seeds = list(nb)
        si = 0
        while si < len(seeds):
            q = seeds[si]
            si += 1
            if not visited[q]:
                visited[q] = True
                q_nb = neighbours(q)
                if len(q_nb) >= min_samples:
                    for nb_q in q_nb:
                        if nb_q not in seeds:
                            seeds.append(nb_q)
            if labels[q] == -1:
                labels[q] = cluster_id

        cluster_id += 1

    return labels, eps_array


# ──────────────────────────────────────────────
#  Wrapper matching clustering.py interface
# ──────────────────────────────────────────────

def cluster_strokes_adaptive(
    strokes: List[Stroke],
    alpha: float = 1.5,
    min_samples: int = 1
) -> Tuple[ClusterMap, np.ndarray, np.ndarray]:
    """
    Drop-in replacement for clustering.cluster_strokes(),
    but uses adaptive per-stroke epsilon.

    Returns
    -------
    clusters  : dict cluster_id → [Stroke, ...]
    labels    : raw label array
    eps_array : per-stroke epsilon values
    """
    t0 = time.perf_counter()
    labels, eps_array = adaptive_dbscan(strokes, alpha=alpha, min_samples=min_samples)
    elapsed = time.perf_counter() - t0

    n_clusters = len(set(labels) - {-1})
    n_noise    = int(np.sum(labels == -1))

    print(f"[AdaptiveCluster] {n_clusters} clusters, {n_noise} noise, "
          f"α={alpha}, time={elapsed*1000:.2f}ms")
    print(f"  ε range: {eps_array.min():.1f}px – {eps_array.max():.1f}px "
          f"(mean {eps_array.mean():.1f}px)")

    clusters: ClusterMap = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(strokes[i])

    return clusters, labels, eps_array