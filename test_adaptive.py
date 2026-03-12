"""
test_adaptive.py — Prove adaptive ε beats fixed ε on scale-variant handwriting

Scenario mirrors the real problem:
  - 2 words written LARGE  (strokes ~150px tall, spread ~300px apart)
  - 2 words written MEDIUM (strokes ~50px tall,  spread ~100px apart)
  - 1 word  written TINY   (strokes ~10px tall,  spread ~20px  apart)

Ground truth: 5 separate word-clusters.

Fixed ε = 80px   → expected to fail  (merges tiny, fragments large)
Adaptive ε       → expected to work  (finds all 5 correctly)
"""

import random
import numpy as np
from stroke import Stroke, preprocess
from clustering import cluster_strokes
from adaptive_clustering import cluster_strokes_adaptive


# ──────────────────────────────────────────────
#  Synthetic scale-variant data generator
# ──────────────────────────────────────────────

def make_word(
    word_id: int,
    n_strokes: int,
    canvas_x: float,
    canvas_y: float,
    char_size: float,       # approximate px height of one character
    char_spacing: float,    # px between character stroke centers
    seed: int = 0
) -> list[Stroke]:
    """
    Simulate one handwritten word as a cluster of strokes.
    Each stroke represents one character (simplified).
    char_size controls how big the strokes are.
    char_spacing controls inter-stroke distance within the word.
    """
    rng = random.Random(seed + word_id * 17)
    strokes = []

    for i in range(n_strokes):
        # Place each character stroke along x axis
        cx = canvas_x + i * char_spacing + rng.uniform(-char_spacing*0.1, char_spacing*0.1)
        cy = canvas_y + rng.uniform(-char_size * 0.1, char_size * 0.1)

        # Generate a stroke whose bounding box is roughly char_size × char_size
        pts = []
        for _ in range(12):
            px = cx + rng.uniform(-char_size/2, char_size/2)
            py = cy + rng.uniform(-char_size/2, char_size/2)
            pts.append((px, py))

        strokes.append(Stroke(
            points=pts,
            timestamp=float(word_id * 100 + i),
            stroke_id=len(strokes) + word_id * 100,
            label=word_id     # ground truth
        ))

    return strokes


def build_test_canvas():
    """
    5 words at very different scales, well separated spatially.
    Returns list of strokes with .label = word index (ground truth).
    """
    all_strokes = []

    # Word 0 — LARGE  (top-left)
    all_strokes += make_word(0, n_strokes=4, canvas_x=200,  canvas_y=200,
                              char_size=150, char_spacing=180, seed=1)

    # Word 1 — LARGE  (top-right, well separated from word 0)
    all_strokes += make_word(1, n_strokes=4, canvas_x=1200, canvas_y=200,
                              char_size=150, char_spacing=180, seed=2)

    # Word 2 — MEDIUM (middle)
    all_strokes += make_word(2, n_strokes=4, canvas_x=500,  canvas_y=700,
                              char_size=50,  char_spacing=60,  seed=3)

    # Word 3 — MEDIUM (middle-right)
    all_strokes += make_word(3, n_strokes=4, canvas_x=900,  canvas_y=700,
                              char_size=50,  char_spacing=60,  seed=4)

    # Word 4 — TINY   (bottom, very small writing)
    all_strokes += make_word(4, n_strokes=4, canvas_x=600,  canvas_y=1100,
                              char_size=10,  char_spacing=14,  seed=5)

    return all_strokes


# ──────────────────────────────────────────────
#  Evaluation helper (same as run_pipeline.py)
# ──────────────────────────────────────────────

def evaluate(strokes, labels):
    from collections import Counter
    true_labels = np.array([s.label for s in strokes])
    precisions, recalls = [], []

    for pred_id in set(labels):
        if pred_id == -1:
            continue
        mask = labels == pred_id
        true_in = true_labels[mask]
        valid = true_in[true_in != -1]
        if len(valid) == 0:
            continue
        maj_label, maj_count = Counter(valid).most_common(1)[0]
        precision = maj_count / len(true_in)
        total = int(np.sum(true_labels == maj_label))
        recall = maj_count / total if total > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)

    P = np.mean(precisions) if precisions else 0.0
    R = np.mean(recalls)    if recalls    else 0.0
    F1 = 2*P*R/(P+R+1e-9)
    return {"P": P, "R": R, "F1": F1,
            "n_pred": len(set(labels)-{-1}),
            "n_true": len(set(np.array([s.label for s in strokes]))-{-1})}


# ──────────────────────────────────────────────
#  Main comparison
# ──────────────────────────────────────────────

def main():
    print("=" * 58)
    print("  Scale-Variant Handwriting — Fixed ε vs Adaptive ε")
    print("=" * 58)
    print()
    print("Canvas setup:")
    print("  Word 0,1 — LARGE  (char_size=150px, spacing=180px)")
    print("  Word 2,3 — MEDIUM (char_size= 50px, spacing= 60px)")
    print("  Word 4   — TINY   (char_size= 10px, spacing= 14px)")
    print("  Ground truth: 5 distinct word-clusters")
    print()

    # Build canvas
    strokes_raw = build_test_canvas()
    strokes = [preprocess(s, n_points=32) for s in strokes_raw]
    print(f"Total strokes: {len(strokes)}\n")

    # ── TEST 1: Fixed ε = 80px ──
    print("─" * 58)
    print("Fixed ε DBSCAN  (ε = 80px)")
    print("─" * 58)
    _, labels_fixed = cluster_strokes(strokes, eps=80.0, min_samples=1)
    m_fixed = evaluate(strokes, labels_fixed)
    print(f"  Predicted clusters : {m_fixed['n_pred']}  (truth = {m_fixed['n_true']})")
    print(f"  Precision : {m_fixed['P']:.4f}")
    print(f"  Recall    : {m_fixed['R']:.4f}")
    print(f"  F1 Score  : {m_fixed['F1']:.4f}")
    verdict_fixed = "✓ OK" if m_fixed['F1'] > 0.9 else "✗ FAILED"
    print(f"  Verdict   : {verdict_fixed}")
    print()

    # ── TEST 2: Fixed ε = 200px (generous) ──
    print("─" * 58)
    print("Fixed ε DBSCAN  (ε = 200px, generous)")
    print("─" * 58)
    _, labels_big = cluster_strokes(strokes, eps=200.0, min_samples=1)
    m_big = evaluate(strokes, labels_big)
    print(f"  Predicted clusters : {m_big['n_pred']}  (truth = {m_big['n_true']})")
    print(f"  Precision : {m_big['P']:.4f}")
    print(f"  Recall    : {m_big['R']:.4f}")
    print(f"  F1 Score  : {m_big['F1']:.4f}")
    verdict_big = "✓ OK" if m_big['F1'] > 0.9 else "✗ FAILED"
    print(f"  Verdict   : {verdict_big}")
    print()

    # ── TEST 3: Adaptive ε (α = 2.5) ──
    print("─" * 58)
    print("Adaptive ε DBSCAN  (α = 2.5)  ← Your contribution")
    print("─" * 58)
    _, labels_adap, eps_arr = cluster_strokes_adaptive(strokes, alpha=1.5, min_samples=1)
    m_adap = evaluate(strokes, labels_adap)
    print(f"  Predicted clusters : {m_adap['n_pred']}  (truth = {m_adap['n_true']})")
    print(f"  Precision : {m_adap['P']:.4f}")
    print(f"  Recall    : {m_adap['R']:.4f}")
    print(f"  F1 Score  : {m_adap['F1']:.4f}")
    verdict_adap = "✓ OK" if m_adap['F1'] > 0.9 else "✗ FAILED"
    print(f"  Verdict   : {verdict_adap}")
    print()

    # ── SUMMARY TABLE ──
    print("=" * 58)
    print(f"  {'Method':<28} {'F1':>6}  {'Clusters':>8}  {'Result'}")
    print(f"  {'─'*28} {'─'*6}  {'─'*8}  {'─'*8}")
    print(f"  {'Fixed ε = 80px':<28} {m_fixed['F1']:>6.4f}  {m_fixed['n_pred']:>8}  {verdict_fixed}")
    print(f"  {'Fixed ε = 200px':<28} {m_big['F1']:>6.4f}  {m_big['n_pred']:>8}  {verdict_big}")
    print(f"  {'Adaptive ε (α=1.5, sym=min)':<28} {m_adap['F1']:>6.4f}  {m_adap['n_pred']:>8}  {verdict_adap}")
    print("=" * 58)
    print()

    improvement = m_adap['F1'] - max(m_fixed['F1'], m_big['F1'])
    if improvement > 0:
        print(f"  Adaptive ε improves F1 by +{improvement:.4f} over best fixed ε")
    print()
    print("  This table is Table 2 in your paper. ↑")
    print("=" * 58)


if __name__ == "__main__":
    main()