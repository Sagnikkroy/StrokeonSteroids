"""
clustering.py — DBSCAN stroke clustering with KD-tree acceleration
Paper: Efficient Spatial Clustering and Recognition of Digital Ink
       in Infinite Canvas Interfaces

ALGORITHM CHOICE — WHY DBSCAN:
  • No need to specify number of clusters in advance (unknown on infinite canvas)
  • Handles noise / isolated strokes gracefully (label = -1)
  • Naturally finds arbitrarily-shaped spatial groups
  • eps and min_samples are intuitive to tune (pixel distance + stroke count)

PARAMETERS TO TUNE:
  eps          — max pixel distance between stroke centers to be "neighbours"
                 Typical starting value: 80–120 px at 96 DPI screen resolution
  min_samples  — minimum strokes to form a cluster core
                 Set to 1 so even single strokes get a cluster label
"""

from typing import List, Dict, Tuple
import numpy as np
from sklearn.cluster import DBSCAN

from stroke import Stroke
from spatial_index import StrokeSpatialIndex


ClusterMap = Dict[int, List[Stroke]]   # cluster_id → list of strokes


def cluster_strokes(
    strokes: List[Stroke],
    eps: float = 80.0,
    min_samples: int = 1
) -> Tuple[ClusterMap, np.ndarray]:
    """
    Group strokes into spatial clusters using DBSCAN.

    Parameters
    ----------
    strokes     : preprocessed list of Stroke objects
    eps         : neighbourhood radius in canvas pixels
    min_samples : minimum strokes in a core neighbourhood

    Returns
    -------
    clusters    : dict mapping cluster_id → [Stroke, ...]
                  Noise strokes (label -1) are excluded.
    labels      : raw label array, length = len(strokes)
    """
    if len(strokes) == 0:
        return {}, np.array([])

    # Build KD-tree index (this is the O(n log n) step)
    index = StrokeSpatialIndex(strokes)
    centers = index.centers       # shape (n, 2)

    # DBSCAN uses the KD-tree internally when metric='euclidean'
    db = DBSCAN(eps=eps, min_samples=min_samples, algorithm='kd_tree',
                metric='euclidean')
    labels = db.fit_predict(centers)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int(np.sum(labels == -1))
    print(f"[Clustering] {n_clusters} clusters found, "
          f"{n_noise} noise strokes, "
          f"eps={eps}px, min_samples={min_samples}")

    # Build cluster map
    clusters: ClusterMap = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue                      # noise — skip
        clusters.setdefault(label, []).append(strokes[i])

    return clusters, labels


def cluster_stats(clusters: ClusterMap) -> None:
    """Print a readable summary of cluster sizes."""
    print(f"\n{'─'*40}")
    print(f"  Total clusters : {len(clusters)}")
    sizes = [len(v) for v in clusters.values()]
    if sizes:
        print(f"  Strokes/cluster: min={min(sizes)}, "
              f"max={max(sizes)}, mean={np.mean(sizes):.1f}")
    print(f"{'─'*40}\n")