"""
InkCluster Inference Server
────────────────────────────
Loads your .pt model locally and serves predictions over HTTP.
The canvas sends stroke data → gets back recognised text.

Usage:
    pip install flask flask-cors torch
    python server.py --model ink_model_best.pt --vocab ink_vocab.json

Then open ink-canvas-hierarchical.html normally in your browser.
"""

import argparse, json, math
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow requests from file:// and localhost

# ── Global model state ──
model     = None
vocab     = None
blank_idx = None
device    = torch.device('cpu')

MAX_STROKES = 32
MAX_PTS     = 64


# ────────────────────────────────────────────────
# Model definition — must match your training code
# ────────────────────────────────────────────────

def make_pos_enc(length, d_model, dev):
    pe  = torch.zeros(length, d_model, device=dev)
    pos = torch.arange(length, device=dev).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2, device=dev).float()
                    * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class InkTransformerV3(nn.Module):
    """
    Matches the architecture saved in your .pt file.
    If you get a state_dict mismatch error, paste the error here
    and we'll adjust the layer names.
    """
    def __init__(self, vocab_size, d_model=128,
                 n_stroke_heads=4, n_word_heads=8,
                 n_stroke_layers=2, n_word_layers=4,
                 dropout=0.1):
        super().__init__()
        self.d_model  = d_model
        self.inp_proj = nn.Linear(5, d_model)
        self.inp_norm = nn.LayerNorm(d_model)

        stroke_layer = nn.TransformerEncoderLayer(
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
        stroke_emb = stroke_emb + make_pos_enc(S, self.d_model, stroke_emb.device)
        stroke_pad_mask = torch.zeros(B, S, dtype=torch.bool, device=features.device)
        for i, ns in enumerate(n_strokes):
            stroke_pad_mask[i, ns:] = True
        word_emb = self.word_enc(stroke_emb, src_key_padding_mask=stroke_pad_mask)
        out = self.norm(word_emb)
        out = self.ctc_head(out)
        return torch.log_softmax(out, dim=-1)


# ────────────────────────────────────────────────
# Feature extraction  (mirrors JS canvas code)
# ────────────────────────────────────────────────

def normalise_strokes(strokes):
    all_pts = [p for s in strokes for p in s]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    h  = max(y_max - y_min, 1e-6)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    return [[(( p[0]-cx)/h, (p[1]-cy)/h) for p in s] for s in strokes]


def strokes_to_tensors(strokes_raw):
    """
    strokes_raw: list of lists of [x, y] pairs  (from JSON request)
    returns: features [1,S,P,5], mask [1,S,P], n_strokes [1]
    """
    strokes = strokes_raw[:MAX_STROKES]
    normed  = normalise_strokes(strokes)
    S, P    = len(normed), MAX_PTS

    features = torch.zeros(1, S, P, 5)
    mask     = torch.zeros(1, S, P)

    for si, pts in enumerate(normed):
        n = min(len(pts), P)
        for pi in range(n):
            dx = 0.0 if pi == 0 else pts[pi][0] - pts[pi-1][0]
            dy = 0.0 if pi == 0 else pts[pi][1] - pts[pi-1][1]
            angle = math.atan2(dy, dx)
            pen   = 1.0 if pi == n-1 else 0.0
            features[0, si, pi] = torch.tensor(
                [dx, dy, math.sin(angle), math.cos(angle), pen])
            mask[0, si, pi] = 1.0

    n_strokes = torch.tensor([S])
    return features, mask, n_strokes


# ────────────────────────────────────────────────
# CTC decode
# ────────────────────────────────────────────────

def ctc_greedy_decode(log_probs_np, n_strokes_val):
    """log_probs_np: numpy [S, V]"""
    T   = n_strokes_val
    idx = log_probs_np[:T].argmax(axis=1).tolist()
    # collapse repeats
    collapsed, prev = [], None
    for i in idx:
        if i != prev:
            collapsed.append(i)
            prev = i
    # remove blank and pad
    tokens = [i for i in collapsed if i != blank_idx and i != 0]
    return ''.join(vocab[i] for i in tokens if i < len(vocab))


# ────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'vocab_size': len(vocab) if vocab else 0,
    })


@app.route('/recognise', methods=['POST'])
def recognise():
    """
    POST JSON:
    {
      "strokes": [[[x,y],[x,y],...], [[x,y],...], ...]
    }

    Returns:
    { "text": "hello" }
    """
    if model is None:
        return jsonify({'error': 'model not loaded'}), 503

    data    = request.get_json()
    strokes = data.get('strokes', [])

    if not strokes:
        return jsonify({'text': ''})

    try:
        features, mask, n_strokes = strokes_to_tensors(strokes)
        with torch.no_grad():
            log_probs = model(features, mask, n_strokes)  # [1, S, V]
        text = ctc_greedy_decode(
            log_probs[0].numpy(),
            n_strokes[0].item()
        )
        return jsonify({'text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/recognise_batch', methods=['POST'])
def recognise_batch():
    """
    POST JSON:
    {
      "words":   [{"id":"w0","strokes":[...]}, ...],
      "chars":   [{"id":"c0","strokes":[...]}, ...],
      "phrase_word_ids": {"p0":["w0","w1"], "p1":["w2"]}
    }

    Returns:
    { "results": {"w0":"hello","c0":"h","p0":"hello world"} }
    """
    if model is None:
        return jsonify({'error': 'model not loaded'}), 503

    data    = request.get_json()
    results = {}

    # ── Recognise words (main inference) ──
    for w in data.get('words', []):
        strokes = w['strokes']
        if not strokes:
            results[w['id']] = ''
            continue
        try:
            features, mask, n_strokes = strokes_to_tensors(strokes)
            with torch.no_grad():
                log_probs = model(features, mask, n_strokes)
            results[w['id']] = ctc_greedy_decode(log_probs[0].numpy(), n_strokes[0].item())
        except Exception as e:
            results[w['id']] = ''

    # ── Recognise chars ──
    for c in data.get('chars', []):
        strokes = c['strokes']
        if not strokes:
            results[c['id']] = ''
            continue
        try:
            features, mask, n_strokes = strokes_to_tensors(strokes)
            with torch.no_grad():
                log_probs = model(features, mask, n_strokes)
            results[c['id']] = ctc_greedy_decode(log_probs[0].numpy(), n_strokes[0].item())
        except Exception as e:
            results[c['id']] = ''

    # ── Assemble phrase text from its words (no separate inference) ──
    for pid, wids in data.get('phrase_word_ids', {}).items():
        words = [results.get(wid, '') for wid in wids]
        results[pid] = ' '.join(w for w in words if w)

    return jsonify({'results': results})


# ────────────────────────────────────────────────
# Startup
# ────────────────────────────────────────────────

def load_model(pt_path, vocab_path):
    global model, vocab, blank_idx

    print(f'Loading model: {pt_path}')
    ckpt = torch.load(pt_path, map_location='cpu')
    sd   = ckpt['model_state']

    # ── Always get vocab from checkpoint first, fall back to JSON ──
    if 'vocab' in ckpt:
        vocab     = ckpt['vocab']
        blank_idx = ckpt['blank_idx']
        print(f'  Vocab from checkpoint: {len(vocab)} chars')
    else:
        print(f'Loading vocab: {vocab_path}')
        with open(vocab_path, encoding='utf-8') as f:
            data      = json.load(f)
            vocab     = data['vocab']
            blank_idx = data['blank_idx']
        print(f'  Vocab from JSON: {len(vocab)} chars')

    # ── Auto-detect architecture directly from checkpoint shapes ──
    d_model        = sd['inp_proj.weight'].shape[0]
    vocab_size     = sd['ctc_head.weight'].shape[0]  # ground truth from weights
    n_stroke_layers= sum(1 for k in sd if k.startswith('stroke_enc.layers.') and k.endswith('.norm1.weight'))
    n_word_layers  = sum(1 for k in sd if k.startswith('word_enc.layers.')   and k.endswith('.norm1.weight'))
    n_stroke_heads = max(1, d_model // 16)
    n_word_heads   = max(1, d_model // 16)

    print(f'  Detected: d_model={d_model}, vocab={vocab_size}, '
          f'stroke_layers={n_stroke_layers}, word_layers={n_word_layers}')

    model = InkTransformerV3(
        vocab_size      = vocab_size,
        d_model         = d_model,
        n_stroke_heads  = n_stroke_heads,
        n_word_heads    = n_word_heads,
        n_stroke_layers = n_stroke_layers,
        n_word_layers   = n_word_layers,
    )
    model.load_state_dict(sd)
    model.eval()
    print(f'  Model loaded OK')

    # Quick test forward pass
    with torch.no_grad():
        test_out = model(
            torch.zeros(1, 4, MAX_PTS, 5),
            torch.ones(1, 4, MAX_PTS),
            torch.tensor([4])
        )
    print(f'  Test forward pass OK — output: {test_out.shape}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ink_model_best.pt',
                        help='Path to .pt checkpoint')
    parser.add_argument('--vocab', default='ink_vocab.json',
                        help='Path to vocab JSON')
    parser.add_argument('--port', type=int, default=5001,
                        help='Port to serve on')
    args = parser.parse_args()

    load_model(args.model, args.vocab)
    print(f'\n✓ Server running at http://localhost:{args.port}')
    print(f'  Health check: http://localhost:{args.port}/health\n')
    app.run(host='0.0.0.0', port=args.port, debug=False)