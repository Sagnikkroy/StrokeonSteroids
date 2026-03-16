"""
InkCluster — Full Training Pipeline
=====================================
One file. Run it. Everything downloads and saves to your HDD.

Usage:
    python train_full.py --hdd E:
    python train_full.py --hdd F: --epochs 100 --batch 32

Requirements:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install editdistance tqdm requests
"""

import os, sys, math, tarfile, glob, random, argparse, json, time
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

try:
    import editdistance
except ImportError:
    os.system('pip install editdistance')
    import editdistance

# ══════════════════════════════════════════════════════════════
# 0. ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description='InkCluster Training')
parser.add_argument('--hdd',       type=str, default='E:',  help='HDD drive letter e.g. E:')
parser.add_argument('--epochs',    type=int, default=100,   help='Number of training epochs')
parser.add_argument('--batch',     type=int, default=32,    help='Batch size')
parser.add_argument('--lr',        type=float, default=3e-4,help='Learning rate')
parser.add_argument('--d_model',   type=int, default=64,    help='Model dimension')
parser.add_argument('--workers',   type=int, default=0,     help='DataLoader workers (0 = main process, safest on Windows)')
parser.add_argument('--excerpt',   action='store_true',     help='Use excerpt instead of full dataset (for testing)')
parser.add_argument('--resume',    action='store_true',     help='Resume from last checkpoint if exists')
args = parser.parse_args()

# ══════════════════════════════════════════════════════════════
# 1. PATHS — ALL ON HDD
# ══════════════════════════════════════════════════════════════

HDD        = args.hdd.rstrip('/').rstrip('\\')
BASE_DIR   = os.path.join(HDD, 'inkcluster')
DATA_DIR   = os.path.join(BASE_DIR, 'data', 'mathwriting')
CKPT_DIR   = os.path.join(BASE_DIR, 'checkpoints')
LOG_DIR    = os.path.join(BASE_DIR, 'logs')
BEST_PT    = os.path.join(CKPT_DIR, 'ink_model_best.pt')
LAST_PT    = os.path.join(CKPT_DIR, 'ink_model_last.pt')
VOCAB_JSON = os.path.join(CKPT_DIR, 'ink_vocab.json')
TGZFILE    = os.path.join(BASE_DIR, 'mathwriting.tgz')
LOG_FILE   = os.path.join(LOG_DIR,  'training_log.txt')

for d in [DATA_DIR, CKPT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

print(f'\n{"="*60}')
print(f'  InkCluster Training Pipeline')
print(f'{"="*60}')
print(f'  HDD base     : {BASE_DIR}')
print(f'  Dataset      : {DATA_DIR}')
print(f'  Checkpoints  : {CKPT_DIR}')
print(f'  Epochs       : {args.epochs}')
print(f'  Batch size   : {args.batch}')
print(f'  d_model      : {args.d_model}')
print(f'{"="*60}\n')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

# ══════════════════════════════════════════════════════════════
# 2. DOWNLOAD DATASET TO HDD
# ══════════════════════════════════════════════════════════════

if args.excerpt:
    URL = 'https://storage.googleapis.com/mathwriting_data/mathwriting-2024-excerpt.tgz'
    print('Mode: EXCERPT (small, for testing)')
else:
    URL = 'https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz'
    print('Mode: FULL DATASET (2.9 GB)')

# Check if already downloaded and extracted
# Check if already downloaded and extracted
all_inkml = glob.glob(os.path.join(DATA_DIR, '**', '*.inkml'), recursive=True)
if not all_inkml:
    all_inkml = glob.glob(os.path.join(HDD, 'inkcluster', 'data', '**', '*.inkml'), recursive=True)

if len(all_inkml) > 0:
    print(f'Dataset already extracted: {len(all_inkml)} files found. Skipping download+extract.')
else:
    if not os.path.exists(TGZFILE):
        print(f'\nDownloading to {TGZFILE} ...')
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                pct = count * block_size * 100 / total_size
                done = int(pct / 2)
                mb   = count * block_size / 1e6
                total_mb = total_size / 1e6
                sys.stdout.write(f'\r  [{("█"*done).ljust(50)}] {pct:.1f}%  {mb:.0f}/{total_mb:.0f} MB')
                sys.stdout.flush()
        urllib.request.urlretrieve(URL, TGZFILE, reporthook)
        print(f'\n  Downloaded: {os.path.getsize(TGZFILE)/1e9:.2f} GB')
    else:
        print(f'Archive already on HDD: {TGZFILE}')

    print(f'Extracting to {DATA_DIR} ...')
    with tarfile.open(TGZFILE, 'r:gz') as tar:
        members = tar.getmembers()
        for i, m in enumerate(tqdm(members, desc='Extracting')):
            tar.extract(m, os.path.join(HDD, 'inkcluster', 'data'))
    print('Extraction complete.')

    all_inkml = glob.glob(os.path.join(DATA_DIR, '**', '*.inkml'), recursive=True)
    if not all_inkml:
        all_inkml = glob.glob(os.path.join(HDD, 'inkcluster', '**', '*.inkml'), recursive=True)
    print(f'Found {len(all_inkml)} InkML files.')

if not all_inkml:
    print('ERROR: No .inkml files found. Check extraction.')
    sys.exit(1)

print(f'Total InkML files: {len(all_inkml)}')

# ══════════════════════════════════════════════════════════════
# 3. PARSE InkML
# ══════════════════════════════════════════════════════════════

NS = {'ink': 'http://www.w3.org/2003/InkML'}

def parse_inkml(filepath):
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        # ── Label: type="label" not type="truth" ──
        label = ''
        for ann in root.findall('ink:annotation', NS):
            if ann.get('type') == 'label':
                label = (ann.text or '').strip()
                break

        if not label:
            return [], ''

        # ── Traces: each point is "X Y T" space-separated, points comma-separated ──
        strokes = []
        for trace in root.findall('ink:trace', NS):
            text = (trace.text or '').strip()
            if not text:
                continue
            points = []
            for pt in text.split(','):
                vals = pt.strip().split()
                if len(vals) >= 2:
                    try:
                        points.append([float(vals[0]), float(vals[1])])  # X Y only, ignore T
                    except ValueError:
                        continue
            if len(points) >= 2:
                strokes.append(np.array(points, dtype=np.float32))

        return strokes, label
    except:
        return [], ''

# ══════════════════════════════════════════════════════════════
# 4. BUILD VOCABULARY
# ══════════════════════════════════════════════════════════════

print('\nBuilding vocabulary...')
valid_files, all_labels = [], []

if os.path.exists(VOCAB_JSON):
    print(f'Vocab already exists: {VOCAB_JSON} — loading...')
    with open(VOCAB_JSON) as f:
        vdata = json.load(f)
    VOCAB      = vdata['vocab']
    BLANK_IDX  = vdata['blank_idx']
    PAD_IDX    = 0
    VOCAB_SIZE = len(VOCAB)
    char2idx   = {c: i for i, c in enumerate(VOCAB)}
    idx2char   = {i: c for i, c in enumerate(VOCAB)}
    print(f'Vocab loaded: {VOCAB_SIZE} chars')

    # Still need valid_files — scan without parsing labels
    print('Scanning for valid files...')
    for fp in tqdm(all_inkml, desc='Scanning'):
        s, l = parse_inkml(fp)
        if len(s) >= 1 and len(l) >= 1:
            valid_files.append(fp)
    print(f'Valid samples: {len(valid_files)}')

else:
    for fp in tqdm(all_inkml, desc='Scanning labels'):
        s, l = parse_inkml(fp)
        if len(s) >= 1 and len(l) >= 1:
            valid_files.append(fp)
            all_labels.append(l)

    print(f'Valid samples: {len(valid_files)} / {len(all_inkml)}')
    all_chars  = sorted(set(''.join(all_labels)))
    VOCAB      = ['<PAD>'] + all_chars + ['<BLANK>']
    BLANK_IDX  = len(VOCAB) - 1
    PAD_IDX    = 0
    VOCAB_SIZE = len(VOCAB)
    char2idx   = {c: i for i, c in enumerate(VOCAB)}
    idx2char   = {i: c for i, c in enumerate(VOCAB)}
    print(f'Vocabulary size: {VOCAB_SIZE}')
    print(f'Sample chars: {all_chars[:30]}')

    with open(VOCAB_JSON, 'w') as f:
        json.dump({'vocab': VOCAB, 'blank_idx': BLANK_IDX}, f)
    print(f'Vocab saved: {VOCAB_JSON}')

def encode_label(text):
    return [char2idx[c] for c in text if c in char2idx]

def decode_label(indices):
    return ''.join(idx2char[i] for i in indices
                   if i not in (PAD_IDX, BLANK_IDX) and i in idx2char)

# ══════════════════════════════════════════════════════════════
# 5. FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════

MAX_STROKES = 32
MAX_PTS     = 64
MAX_LABEL   = 128

def normalise_strokes(strokes):
    all_pts = np.concatenate(strokes, axis=0)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)
    height = max(y_max - y_min, 1e-6)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    normed = []
    for s in strokes:
        sn = s.copy()
        sn[:, 0] = (sn[:, 0] - cx) / height
        sn[:, 1] = (sn[:, 1] - cy) / height
        normed.append(sn)
    return normed

def strokes_to_features(strokes, max_pts=MAX_PTS):
    if not strokes: return None, None
    strokes = normalise_strokes(strokes)
    n = len(strokes)
    features = np.zeros((n, max_pts, 5), dtype=np.float32)
    mask     = np.zeros((n, max_pts),    dtype=np.float32)
    for si, stroke in enumerate(strokes):
        pts = stroke[:max_pts]
        for pi in range(len(pts)):
            dx = 0.0 if pi == 0 else float(pts[pi,0] - pts[pi-1,0])
            dy = 0.0 if pi == 0 else float(pts[pi,1] - pts[pi-1,1])
            angle = math.atan2(dy, dx)
            pen   = 1.0 if pi == len(pts)-1 else 0.0
            features[si, pi] = [dx, dy, math.sin(angle), math.cos(angle), pen]
            mask[si, pi] = 1.0
    return features, mask

# ══════════════════════════════════════════════════════════════
# 6. DATASET
# ══════════════════════════════════════════════════════════════

class InkDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        strokes, label = parse_inkml(self.files[idx])
        strokes = strokes[:MAX_STROKES]
        feats, mask = strokes_to_features(strokes)
        if feats is None:
            feats = np.zeros((1, MAX_PTS, 5), dtype=np.float32)
            mask  = np.zeros((1, MAX_PTS),    dtype=np.float32)
        label_enc = encode_label(label)[:MAX_LABEL]
        return {
            'features':    torch.FloatTensor(feats),
            'stroke_mask': torch.FloatTensor(mask),
            'label':       torch.LongTensor(label_enc),
            'n_strokes':   len(strokes),
            'label_len':   len(label_enc),
            'label_str':   label,
        }

def collate_fn(batch):
    max_s = max(b['n_strokes'] for b in batch)
    max_l = max(b['label_len'] for b in batch)
    B = len(batch)
    features    = torch.zeros(B, max_s, MAX_PTS, 5)
    stroke_mask = torch.zeros(B, max_s, MAX_PTS)
    labels      = torch.zeros(B, max(max_l,1), dtype=torch.long)
    n_strokes   = torch.zeros(B, dtype=torch.long)
    label_lens  = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        s, l = b['n_strokes'], b['label_len']
        features[i, :s]    = b['features']
        stroke_mask[i, :s] = b['stroke_mask']
        if l > 0: labels[i, :l] = b['label']
        n_strokes[i]  = s
        label_lens[i] = l
    return {
        'features':    features,
        'stroke_mask': stroke_mask,
        'labels':      labels,
        'n_strokes':   n_strokes,
        'label_lens':  label_lens,
        'label_strs':  [b['label_str'] for b in batch],
    }

# ══════════════════════════════════════════════════════════════
# 7. MODEL DEFINITION (module-level so DataLoader workers can pickle it)
# ══════════════════════════════════════════════════════════════

def make_pos_enc(length, d_model, dev):
    pe  = torch.zeros(length, d_model, device=dev)
    pos = torch.arange(length, device=dev).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2, device=dev).float()
                    * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)

class InkTransformerV3(nn.Module):
    def __init__(self, vocab_size, d_model=64,
                 n_stroke_heads=4, n_word_heads=4,
                 n_stroke_layers=2, n_word_layers=2,
                 dropout=0.1):
        super().__init__()
        self.d_model  = d_model
        self.inp_proj = nn.Linear(5, d_model)
        self.inp_norm = nn.LayerNorm(d_model)
        stroke_layer  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_stroke_heads,
            dim_feedforward=d_model*2, dropout=dropout,
            batch_first=True, norm_first=True)
        self.stroke_enc = nn.TransformerEncoder(stroke_layer, n_stroke_layers)
        word_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_word_heads,
            dim_feedforward=d_model*4, dropout=dropout,
            batch_first=True, norm_first=True)
        self.word_enc = nn.TransformerEncoder(word_layer, n_word_layers)
        self.norm     = nn.LayerNorm(d_model)
        self.ctc_head = nn.Linear(d_model, vocab_size)
        self.dropout  = nn.Dropout(dropout)

    def encode_strokes(self, x, stroke_mask):
        B, S, P, F = x.shape
        x    = x.view(B*S, P, F)
        mask = stroke_mask.view(B*S, P)
        x    = self.inp_proj(x)
        x    = self.inp_norm(x)
        x    = x + make_pos_enc(P, self.d_model, x.device)
        key_mask = (mask == 0)
        x    = self.stroke_enc(x, src_key_padding_mask=key_mask)
        mask_exp = mask.unsqueeze(-1)
        x    = (x * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1)
        return x.view(B, S, self.d_model)

    def forward(self, features, stroke_mask, n_strokes):
        B, S = features.shape[:2]
        stroke_emb = self.encode_strokes(features, stroke_mask)
        stroke_emb = self.dropout(stroke_emb)
        stroke_emb = stroke_emb + make_pos_enc(S, self.d_model, stroke_emb.device)
        stroke_pad_mask = torch.zeros(B, S, dtype=torch.bool, device=features.device)
        for i, ns in enumerate(n_strokes):
            stroke_pad_mask[i, ns:] = True
        word_emb = self.word_enc(stroke_emb, src_key_padding_mask=stroke_pad_mask)
        out = self.norm(word_emb)
        out = self.ctc_head(out)
        return torch.log_softmax(out, dim=-1)

if __name__ == '__main__':
    random.shuffle(valid_files)
    split      = int(0.9 * len(valid_files))
    train_ds   = InkDataset(valid_files[:split])
    val_ds     = InkDataset(valid_files[split:])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.workers,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              collate_fn=collate_fn, num_workers=args.workers,
                              pin_memory=torch.cuda.is_available())

    print(f'\nTrain: {len(train_ds)} | Val: {len(val_ds)}')
    print(f'Train batches: {len(train_loader)}')

    model = InkTransformerV3(
        vocab_size     = VOCAB_SIZE,
        d_model        = args.d_model,
        n_stroke_heads = max(1, args.d_model // 16),
        n_word_heads   = max(1, args.d_model // 16),
        n_stroke_layers= 2,
        n_word_layers  = 2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel parameters: {total_params:,} (~{total_params/1e6:.2f}M)')

    # ── CTC decode ──
    def ctc_greedy_decode(log_probs, n):
        indices = log_probs[:n].argmax(dim=-1).tolist()
        collapsed, prev = [], None
        for i in indices:
            if i != prev: collapsed.append(i); prev = i
        return decode_label([i for i in collapsed if i != BLANK_IDX and i != PAD_IDX])

    def decode_batch(lp, ns):
        return [ctc_greedy_decode(lp[i], ns[i].item()) for i in range(lp.shape[0])]

    def compute_cer(preds, targets):
        total_d, total_l = 0, 0
        for p, t in zip(preds, targets):
            total_d += editdistance.eval(p, t)
            total_l += max(len(t), 1)
        return total_d / total_l

    # ── Training setup ──
    ctc_loss  = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True, reduction='mean')
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch  = 1
    best_val_cer = float('inf')
    history      = {'train_loss': [], 'val_loss': [], 'val_cer': []}

    if args.resume and os.path.exists(LAST_PT):
        print(f'\nResuming from {LAST_PT}')
        ckpt = torch.load(LAST_PT, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch  = ckpt['epoch'] + 1
        best_val_cer = ckpt.get('best_val_cer', float('inf'))
        history      = ckpt.get('history', history)
        print(f'  Resumed from epoch {ckpt["epoch"]}, best CER={best_val_cer:.4f}')

    # ── Train / val loops ──
    def train_epoch():
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc='  Train', leave=False, ncols=80):
            features    = batch['features'].to(device)
            stroke_mask = batch['stroke_mask'].to(device)
            labels      = batch['labels'].to(device)
            n_strokes   = batch['n_strokes'].to(device)
            label_lens  = batch['label_lens'].to(device)
            if label_lens.sum() == 0: continue
            log_probs = model(features, stroke_mask, n_strokes)
            loss = ctc_loss(log_probs.permute(1,0,2), labels, n_strokes, label_lens)
            if torch.isnan(loss) or torch.isinf(loss): continue
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(train_loader), 1)

    @torch.no_grad()
    def val_epoch():
        model.eval()
        total_loss, all_preds, all_tgts = 0, [], []
        for batch in tqdm(val_loader, desc='  Val  ', leave=False, ncols=80):
            features    = batch['features'].to(device)
            stroke_mask = batch['stroke_mask'].to(device)
            labels      = batch['labels'].to(device)
            n_strokes   = batch['n_strokes'].to(device)
            label_lens  = batch['label_lens'].to(device)
            if label_lens.sum() == 0: continue
            log_probs = model(features, stroke_mask, n_strokes)
            loss = ctc_loss(log_probs.permute(1,0,2), labels, n_strokes, label_lens)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
            all_preds.extend(decode_batch(log_probs.cpu(), n_strokes.cpu()))
            all_tgts.extend(batch['label_strs'])
        cer = compute_cer(all_preds, all_tgts)
        return total_loss / max(len(val_loader), 1), cer, all_preds[:4], all_tgts[:4]

    def log(msg):
        print(msg)
        with open(LOG_FILE, 'a') as f:
            f.write(msg + '\n')

    # ── Main training loop ──
    log(f'\n{"="*60}')
    log(f'Training started: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    log(f'Device: {device}  |  Epochs: {args.epochs}  |  Batch: {args.batch}')
    log(f'{"="*60}')

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch()
        val_loss, val_cer, sample_preds, sample_tgts = val_epoch()
        scheduler.step()
        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_cer'].append(val_cer)

        # Save last checkpoint always
        torch.save({
            'epoch': epoch, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'vocab': VOCAB, 'blank_idx': BLANK_IDX,
            'val_cer': val_cer, 'best_val_cer': best_val_cer,
            'history': history,
        }, LAST_PT)

        # Save best checkpoint
        saved = ''
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'vocab': VOCAB, 'blank_idx': BLANK_IDX, 'val_cer': val_cer,
            }, BEST_PT)
            saved = ' ← BEST SAVED'

        log(f'Epoch {epoch:3d}/{args.epochs} | loss={train_loss:.4f} | '
            f'val_loss={val_loss:.4f} | CER={val_cer:.4f} | {elapsed:.0f}s{saved}')

        if epoch % 5 == 0:
            log('  Samples:')
            for p, t in zip(sample_preds, sample_tgts):
                log(f'    pred: "{p}"  |  tgt: "{t}"')

        if epoch % 10 == 0:
            with open(os.path.join(LOG_DIR, 'history.json'), 'w') as f:
                json.dump(history, f)

    log(f'\nTraining complete. Best CER: {best_val_cer:.4f}')
    log(f'Best model: {BEST_PT}')
    log(f'Vocab:      {VOCAB_JSON}')