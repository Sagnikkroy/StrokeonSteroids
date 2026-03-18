"""
preprocess.py — Run ONCE before training.
Converts all InkML files to memory-mapped numpy arrays on HDD.

Usage:
    python preprocess.py --hdd E:
"""

import os, sys, glob, math, json, argparse
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--hdd',         type=str, default='E:')
parser.add_argument('--max_strokes', type=int, default=32)
parser.add_argument('--max_pts',     type=int, default=64)
parser.add_argument('--max_label',   type=int, default=128)
args = parser.parse_args()

HDD      = args.hdd.rstrip('/').rstrip('\\')
BASE     = os.path.join(HDD, 'inkcluster')
DATA_DIR = os.path.join(BASE, 'data')
CKPT_DIR = os.path.join(BASE, 'checkpoints')
CACHE    = os.path.join(BASE, 'cache')
VOCAB_JSON = os.path.join(CKPT_DIR, 'ink_vocab.json')
PROGRESS   = os.path.join(CACHE, 'progress.json')

MAX_S = args.max_strokes
MAX_P = args.max_pts
MAX_L = args.max_label

os.makedirs(CACHE, exist_ok=True)

NS = {'ink': 'http://www.w3.org/2003/InkML'}

# ── Parser ──────────────────────────────────────────────────
def parse_inkml(fp):
    try:
        tree = ET.parse(fp)
        root = tree.getroot()
        label = ''
        for ann in root.findall('ink:annotation', NS):
            if ann.get('type') == 'label':
                label = (ann.text or '').strip()
                break
        if not label:
            return [], ''
        strokes = []
        for trace in root.findall('ink:trace', NS):
            text = (trace.text or '').strip()
            if not text:
                continue
            pts = []
            for pt in text.split(','):
                v = pt.strip().split()
                if len(v) >= 2:
                    try:
                        pts.append([float(v[0]), float(v[1])])
                    except ValueError:
                        continue
            if len(pts) >= 2:
                strokes.append(np.array(pts, dtype=np.float32))
        return strokes, label
    except:
        return [], ''

# ── Feature extraction ──────────────────────────────────────
def to_features(raw_strokes):
    """Always returns shape (MAX_S, MAX_P, 5), (MAX_S, MAX_P), int."""
    strokes = raw_strokes[:MAX_S]
    # normalise
    all_pts = np.concatenate(strokes, axis=0)
    xmin, ymin = all_pts.min(0)
    xmax, ymax = all_pts.max(0)
    h  = max(ymax - ymin, 1e-6)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    normed = [np.stack([(s[:,0]-cx)/h, (s[:,1]-cy)/h], axis=1) for s in strokes]

    S    = len(normed)
    feat = np.zeros((MAX_S, MAX_P, 5), dtype=np.float32)   # <-- MAX_S not S
    mask = np.zeros((MAX_S, MAX_P),    dtype=np.float32)   # <-- MAX_S not S

    for si, pts in enumerate(normed):
        n = min(len(pts), MAX_P)
        for pi in range(n):
            dx = 0. if pi == 0 else float(pts[pi,0] - pts[pi-1,0])
            dy = 0. if pi == 0 else float(pts[pi,1] - pts[pi-1,1])
            a  = math.atan2(dy, dx)
            feat[si, pi] = [dx, dy, math.sin(a), math.cos(a), 1. if pi==n-1 else 0.]
            mask[si, pi] = 1.
    return feat, mask, S

# ── Load vocab ──────────────────────────────────────────────
print(f'Loading vocab from {VOCAB_JSON}')
with open(VOCAB_JSON, encoding='utf-8') as f:
    vdata = json.load(f)
VOCAB     = vdata['vocab']
BLANK_IDX = vdata['blank_idx']
char2idx  = {c: i for i, c in enumerate(VOCAB)}
print(f'Vocab: {len(VOCAB)} chars')

# ── Find all files ──────────────────────────────────────────
all_inkml = glob.glob(os.path.join(DATA_DIR, '**', '*.inkml'), recursive=True)
N = len(all_inkml)
print(f'Files: {N:,}')

# ── Create memory-mapped arrays on HDD ──────────────────────
feat_path = os.path.join(CACHE, 'features.npy')
mask_path = os.path.join(CACHE, 'masks.npy')
ns_path   = os.path.join(CACHE, 'nstrokes.npy')
le_path   = os.path.join(CACHE, 'labelenc.npy')
ll_path   = os.path.join(CACHE, 'labellen.npy')
lb_path   = os.path.join(CACHE, 'labels.json')

# Resume support
start  = 0
labels = []
if os.path.exists(PROGRESS):
    with open(PROGRESS) as f:
        prog  = json.load(f)
    start  = prog['done']
    labels = prog['labels']
    print(f'Resuming from {start:,}')
    mode = 'r+'
else:
    mode = 'w+'

feat_mm = np.memmap(feat_path, dtype='float32', mode=mode, shape=(N, MAX_S, MAX_P, 5))
mask_mm = np.memmap(mask_path, dtype='float32', mode=mode, shape=(N, MAX_S, MAX_P))
ns_mm   = np.memmap(ns_path,   dtype='int32',   mode=mode, shape=(N,))
le_mm   = np.memmap(le_path,   dtype='int32',   mode=mode, shape=(N, MAX_L))
ll_mm   = np.memmap(ll_path,   dtype='int32',   mode=mode, shape=(N,))

print(f'\nExtracting features...')
FLUSH = 5000

for i in tqdm(range(start, N), ncols=80, initial=start, total=N):
    strokes, label = parse_inkml(all_inkml[i])

    if len(strokes) < 1 or not label:
        labels.append('')
        continue

    feat, mask, ns = to_features(strokes)          # (MAX_S,MAX_P,5), (MAX_S,MAX_P), int
    enc = [char2idx.get(c, 0) for c in label if c in char2idx][:MAX_L]

    feat_mm[i] = feat                              # shape matches exactly
    mask_mm[i] = mask
    ns_mm[i]   = ns
    le_mm[i, :len(enc)] = enc
    ll_mm[i]   = len(enc)
    labels.append(label)

    if (i + 1) % FLUSH == 0:
        feat_mm.flush(); mask_mm.flush()
        ns_mm.flush();   le_mm.flush(); ll_mm.flush()
        with open(PROGRESS, 'w') as f:
            json.dump({'done': i+1, 'labels': labels}, f)

# Final flush
feat_mm.flush(); mask_mm.flush()
ns_mm.flush();   le_mm.flush(); ll_mm.flush()

with open(lb_path, 'w', encoding='utf-8') as f:
    json.dump(labels, f, ensure_ascii=False)

if os.path.exists(PROGRESS):
    os.remove(PROGRESS)

total = sum(os.path.getsize(p)/1e9 for p in [feat_path,mask_path,ns_path,le_path,ll_path])
print(f'\nDone! Cache size: {total:.1f} GB')
print(f'Location: {CACHE}')
print(f'\nNow train:')
print(f'  python train_full.py --hdd {HDD} --epochs 100 --batch 32 --cached')