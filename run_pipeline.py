"""
run_pipeline.py — End-to-end pipeline demo with synthetic data
Paper: Efficient Spatial Clustering and Recognition of Digital Ink
       in Infinite Canvas Interfaces

Run this to verify everything works:
    python run_pipeline.py

It will:
  1. Generate synthetic strokes placed in known spatial groups
  2. Preprocess each stroke
  3. Build KD-tree spatial index
  4. Cluster with DBSCAN
  5. Evaluate clustering accuracy (precision / recall)
  6. Print a visual summary
"""

import random
import time
import numpy as np

from stroke import Stroke, preprocess
from clustering import cluster_strokes, cluster_stats


# ------------------------------------------------------------------ #
#  Synthetic data generator                                            #
# ------------------------------------------------------------------ #

def make_synthetic_strokes(
    n_groups: int = 8,
    strokes_per_group: int = 3,
    canvas_size: float = 10000.0,
    intra_group_spread: float = 50.0,
    seed: int = 42
) -> list[Stroke]:
    """
    Place `n_groups` character-like clusters randomly on the canvas.
    Each group has `strokes_per_group` strokes close together,
    far from other groups.

    ground-truth label is stored in stroke.label
    """
    rng = random.Random(seed)
    strokes = []
    stroke_id = 0

    # Place group centers far apart (grid-ish with jitter)
    group_centers = []
    margin = canvas_size * 0.1
    for g in range(n_groups):
        cx = rng.uniform(margin, canvas_size - margin)
        cy = rng.uniform(margin, canvas_size - margin)
        group_centers.append((cx, cy))

    for g, (cx, cy) in enumerate(group_centers):
        for _ in range(strokes_per_group):
            # Each stroke: a short squiggle near the group center
            n_pts = rng.randint(5, 15)
            ox = rng.uniform(-intra_group_spread, intra_group_spread)
            oy = rng.uniform(-intra_group_spread, intra_group_spread)
            pts = [
                (cx + ox + rng.uniform(-10, 10),
                 cy + oy + rng.uniform(-10, 10))
                for _ in range(n_pts)
            ]
            s = Stroke(
                points=pts,
                timestamp=time.time() + stroke_id * 0.1,
                stroke_id=stroke_id,
                label=g          # ground truth
            )
            strokes.append(s)
            stroke_id += 1

    rng_global = random.Random(seed + 1)
    # Add a few isolated noise strokes
    for _ in range(3):
        pts = [(rng_global.uniform(0, canvas_size),
                rng_global.uniform(0, canvas_size)) for _ in range(5)]
        s = Stroke(pts, time.time(), stroke_id, label=-1)
        strokes.append(s)
        stroke_id += 1

    return strokes


# ------------------------------------------------------------------ #
#  Evaluation: clustering precision / recall                           #
# ------------------------------------------------------------------ #

def evaluate_clustering(strokes: list[Stroke],
                         labels: np.ndarray) -> dict:
    """
    Compare predicted cluster labels to ground-truth stroke.label.

    We use the Hungarian-style approach:
      For each predicted cluster, find the majority ground-truth label.
      Precision = majority count / cluster size
      Recall    = majority count / total strokes with that true label

    Returns dict with per-cluster stats and macro averages.
    """
    from collections import Counter

    true_labels = np.array([s.label for s in strokes])
    pred_labels = labels

    precisions, recalls = [], []

    for pred_id in set(pred_labels):
        if pred_id == -1:
            continue
        mask = pred_labels == pred_id
        true_in_cluster = true_labels[mask]

        if len(true_in_cluster) == 0:
            continue

        valid = true_in_cluster[true_in_cluster != -1]
        if len(valid) == 0:
            continue
        majority_label, majority_count = Counter(valid).most_common(1)[0]

        precision = majority_count / len(true_in_cluster)

        total_with_true_label = int(np.sum(true_labels == majority_label))
        recall = majority_count / total_with_true_label if total_with_true_label > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    macro_precision = np.mean(precisions) if precisions else 0.0
    macro_recall    = np.mean(recalls)    if recalls    else 0.0
    f1 = (2 * macro_precision * macro_recall /
          (macro_precision + macro_recall + 1e-9))

    return {
        "macro_precision": macro_precision,
        "macro_recall":    macro_recall,
        "f1":              f1,
        "n_pred_clusters": len(set(pred_labels) - {-1}),
        "n_true_clusters": len(set(true_labels) - {-1}),
    }


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    print("=" * 50)
    print("  Digital Ink Pipeline — Smoke Test")
    print("=" * 50)

    # 1. Generate synthetic data
    print("\n[1/4] Generating synthetic strokes...")
    strokes_raw = make_synthetic_strokes(
        n_groups=8,
        strokes_per_group=3,
        canvas_size=10_000.0
    )
    print(f"      {len(strokes_raw)} strokes on a 10,000×10,000 canvas")

    # 2. Preprocess
    print("\n[2/4] Preprocessing (resample + smooth)...")
    t0 = time.perf_counter()
    strokes = [preprocess(s, n_points=32) for s in strokes_raw]
    print(f"      Done in {(time.perf_counter()-t0)*1000:.1f} ms")

    # 3. Cluster
    print("\n[3/4] Clustering with DBSCAN (eps=80px)...")
    t0 = time.perf_counter()
    clusters, labels = cluster_strokes(strokes, eps=80.0, min_samples=1)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"      Done in {elapsed:.2f} ms")
    cluster_stats(clusters)

    # 4. Evaluate
    print("[4/4] Evaluating clustering accuracy...\n")
    metrics = evaluate_clustering(strokes, labels)

    print(f"  Predicted clusters : {metrics['n_pred_clusters']}")
    print(f"  True clusters      : {metrics['n_true_clusters']}")
    print(f"  Macro Precision    : {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall       : {metrics['macro_recall']:.4f}")
    print(f"  F1 Score           : {metrics['f1']:.4f}")

    print("\n" + "=" * 50)
    if metrics['f1'] > 0.9:
        print("  ✓ Pipeline working correctly.")
    else:
        print("  ⚠ F1 low — try adjusting eps value.")
    print("=" * 50)


if __name__ == "__main__":
    main()