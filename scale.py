import time, random, math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

def adaptive_dbscan(pts, alpha=1.5):
    n = len(pts)
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    eps = alpha * dists[:, 1]
    labels = [-1] * n
    cid = 0
    visited = [False] * n
    for i in range(n):
        if visited[i]: continue
        visited[i] = True
        labels[i] = cid
        nb = tree.query_ball_point(pts[i], eps[i])
        stack = [j for j in nb if j != i]
        while stack:
            q = stack.pop()
            if not visited[q]:
                visited[q] = True
                nb2 = tree.query_ball_point(pts[q], eps[q])
                stack.extend(j for j in nb2 if j != q)
            if labels[q] == -1:
                labels[q] = cid
        cid += 1
    return labels

sizes = [10, 50, 100, 200, 500, 1000, 2000, 5000]
kdtree_times, naive_times = [], []

for n in sizes:
    pts = np.random.rand(n, 2) * 1000
    
    # KD-Tree version
    t0 = time.perf_counter()
    for _ in range(5): adaptive_dbscan(pts.tolist())
    kdtree_times.append((time.perf_counter()-t0)/5*1000)
    
    # Naive O(n^2) version
    if n <= 1000:
        t0 = time.perf_counter()
        for _ in range(3):
            dists = [[math.hypot(pts[i][0]-pts[j][0], pts[i][1]-pts[j][1]) for j in range(n)] for i in range(n)]
        naive_times.append((time.perf_counter()-t0)/3*1000)
    else:
        naive_times.append(None)

plt.figure(figsize=(8,5))
plt.plot(sizes, kdtree_times, 'o-', color='#64ffda', linewidth=2, label='Adaptive DBSCAN + KD-Tree (O(n log n))')
valid = [(s,t) for s,t in zip(sizes, naive_times) if t is not None]
plt.plot([s for s,t in valid], [t for s,t in valid], 's--', color='#ff6b6b', linewidth=2, label='Naive O(n²)')
plt.xlabel('Number of Strokes (n)')
plt.ylabel('Time (ms)')
plt.title('Scalability: KD-Tree vs Naive DBSCAN')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('scalability.png', dpi=150)
plt.show()
print('saved scalability.png')