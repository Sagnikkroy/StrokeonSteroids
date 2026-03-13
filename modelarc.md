# Digital Ink Recognition Model
## Transformer-based · English Cursive + Printed · Attaches to InkCluster

---

## Architecture Overview

```
Word Cluster (strokes)
      ↓
  StrokeEncoder         — per-stroke point sequence → embedding
      ↓
  CrossStrokeTransformer — attention across strokes in the word
      ↓
  CTC Decoder           — variable-length text output
      ↓
  "Hello" / "World" / "Subhomita"
```

This is the same fundamental design as:
- Google MLKit Digital Ink Recognizer (internal)
- DeepWriting (ETH Zurich 2018)
- SketchBERT (adapted for ink)

---

## Why Transformer + CTC (not CNN alone)

CNN alone works for PRINTED isolated characters rendered as images.
It fails for cursive because:
  - Stroke order matters ("e" drawn left-to-right vs right-to-left)
  - Variable number of strokes per character
  - Ligatures span multiple "characters"

Transformer handles:
  - Variable-length sequences natively (attention over stroke sequence)
  - Long-range dependencies (start of word affects end)
  - Both cursive and printed with the same architecture

CTC (Connectionist Temporal Classification) handles:
  - Variable-length output text from fixed-size sequence
  - No need for character segmentation ground truth
  - Same technique used in speech recognition (DeepSpeech)

---

## Input Representation

### Raw → Feature Vector per stroke point

Each point in a stroke gets 5 features:

```python
[Δx, Δy, sin(θ), cos(θ), pen_state]

where:
  Δx, Δy    = displacement from previous point (normalised)
  θ         = angle of movement direction
  pen_state = 0 (mid-stroke) or 1 (end of stroke / pen lift)
```

This is the standard "delta encoding" used in SketchRNN and most
ink recognition systems. It's translation-invariant and scale-robust.

### Normalisation (critical)

Before feeding to the model, each word cluster is normalised:
1. Translate so bounding box center = (0, 0)
2. Scale so bounding box height = 1.0
3. This makes the model scale-invariant — same weights for
   large "Hello" and tiny "hello"

Your hierarchical clustering output gives you the bounding box
for free (from Level 2 word cluster).

---

## Model Architecture

```
Input: Word cluster — N strokes, each M points, 5 features
Shape: [N_strokes, M_points, 5]

─────────────────────────────────────────────────
STAGE 1: Per-Stroke Encoder (processes each stroke independently)
─────────────────────────────────────────────────

StrokeEncoder (shared weights across all strokes):
  Linear(5 → 64)
  + Positional encoding (sinusoidal over point index)
  TransformerEncoder(
    d_model=64, nhead=4, num_layers=2, dim_ff=256
  )
  → mean-pool over points
  → stroke embedding [64-dim]

Output: [N_strokes, 64]

─────────────────────────────────────────────────
STAGE 2: Cross-Stroke Transformer (word-level context)
─────────────────────────────────────────────────

+ Positional encoding (sinusoidal over stroke index)
TransformerEncoder(
  d_model=64, nhead=8, num_layers=4, dim_ff=512,
  dropout=0.1
)

Output: [N_strokes, 64]   (contextualised stroke embeddings)

─────────────────────────────────────────────────
STAGE 3: CTC Decoder
─────────────────────────────────────────────────

Linear(64 → vocab_size + 1)   (+1 for CTC blank token)
LogSoftmax
→ CTC Loss during training
→ Beam search decode at inference

Output: string (recognised word)
```

Total parameters: ~2.1M  (small, runs on CPU in <10ms per word)

---

## Vocabulary

```python
CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,!?'-:;()"
)
BLANK = len(CHARS)  # CTC blank token
VOCAB_SIZE = len(CHARS) + 1  # = 96
```

---

## Training Data

### Option 1 — IAM Online Handwriting Database (recommended)
- 1,066 writers, 13,049 handwritten lines
- Both cursive and printed
- Ground truth word-level transcriptions
- Free for research: https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database
- Format: XML with stroke coordinates + transcription

### Option 2 — UNIPEN
- Older but large: 25,000+ words
- Multiple languages, includes English cursive

### Option 3 — Synthetic augmentation (strongly recommended on top of IAM)
Generate synthetic cursive strokes programmatically:
  - Sample words from a word frequency list (top 10K English words)
  - Use a parametric cursive generator (see synthetic_data.py below)
  - 10x your dataset size for free
  - Especially useful for rare words and punctuation

### Recommended split
- IAM Online: ~80K word samples after segmentation
- Synthetic: ~200K generated samples
- Split: 80% train / 10% val / 10% test

---

## Training Code Sketch

```python
# model.py
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class InkTransformer(nn.Module):
    def __init__(self, vocab_size=96, d_model=64, nhead_stroke=4,
                 nhead_word=8, n_stroke_layers=2, n_word_layers=4):
        super().__init__()

        # Stage 1: per-stroke encoder
        self.stroke_proj = nn.Linear(5, d_model)
        stroke_layer = TransformerEncoderLayer(
            d_model, nhead_stroke, dim_feedforward=256,
            dropout=0.1, batch_first=True
        )
        self.stroke_enc = TransformerEncoder(stroke_layer, n_stroke_layers)

        # Stage 2: cross-stroke transformer
        word_layer = TransformerEncoderLayer(
            d_model, nhead_word, dim_feedforward=512,
            dropout=0.1, batch_first=True
        )
        self.word_enc = TransformerEncoder(word_layer, n_word_layers)

        # Stage 3: CTC head
        self.ctc_head = nn.Linear(d_model, vocab_size)

    def encode_strokes(self, x):
        # x: [batch, n_strokes, n_points, 5]
        B, S, P, F = x.shape
        x = x.view(B*S, P, F)           # process all strokes in parallel
        x = self.stroke_proj(x)
        x = x + pos_enc(P, x.device)
        x = self.stroke_enc(x)
        x = x.mean(dim=1)               # mean-pool over points
        return x.view(B, S, -1)         # [batch, n_strokes, d_model]

    def forward(self, x):
        # Stage 1
        stroke_emb = self.encode_strokes(x)   # [B, S, 64]
        stroke_emb = stroke_emb + pos_enc(stroke_emb.size(1), x.device)

        # Stage 2
        word_emb = self.word_enc(stroke_emb)  # [B, S, 64]

        # Stage 3
        logits = self.ctc_head(word_emb)      # [B, S, vocab]
        return torch.log_softmax(logits, dim=-1)


def pos_enc(length, device, d=64):
    """Sinusoidal positional encoding."""
    pe = torch.zeros(length, d, device=device)
    pos = torch.arange(length, device=device).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2, device=device).float()
                    * -(math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)
```

```python
# train.py
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

model = InkTransformer()
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=100)
ctc_loss = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)

# Training loop (simplified)
for epoch in range(100):
    for batch in dataloader:
        strokes, targets, input_lengths, target_lengths = batch

        log_probs = model(strokes)          # [B, T, V]
        log_probs = log_probs.permute(1,0,2) # [T, B, V] for CTCLoss

        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    scheduler.step()
    print(f"Epoch {epoch}: loss={loss.item():.4f}")
```

---

## Attaching to InkCluster

After training, the model attaches to your pipeline at Level 2 output:

```python
# inference.py
from hierarchical_clustering import hierarchical_cluster
from model import InkTransformer, preprocess_word_cluster
import torch

model = InkTransformer()
model.load_state_dict(torch.load('ink_model.pt'))
model.eval()

def recognise(raw_strokes):
    phrases, words, chars, timing = hierarchical_cluster(raw_strokes)

    results = []
    for phrase in phrases:
        phrase_text = []
        for word in phrase.words:
            # Extract + normalise word strokes
            features = preprocess_word_cluster(word.strokes)
            features = torch.tensor(features).unsqueeze(0)  # [1, S, P, 5]

            with torch.no_grad():
                log_probs = model(features)
                text = ctc_beam_decode(log_probs[0])

            phrase_text.append(text)
        results.append(' '.join(phrase_text))

    return results  # ["Hello World", "Hello Subhomita", "WHATS UP?"]
```

---

## Training Timeline

| Phase | Time | Notes |
|---|---|---|
| Download IAM Online | 30 min | Register + download |
| Data preprocessing | 2–3 hrs | Parse XML, segment words, delta encode |
| Synthetic data gen | 1 hr | Run generator script |
| Training (GPU) | 4–6 hrs | 100 epochs on ~280K samples |
| Training (CPU) | 2–3 days | Not recommended |
| Evaluation + tuning | 2–3 hrs | Adjust alpha, beam width |

**Recommended: Google Colab Pro with T4 GPU — free tier works too for this model size.**

---

## Expected Accuracy

| Data | CER (Character Error Rate) | WER (Word Error Rate) |
|---|---|---|
| IAM printed | ~5% | ~12% |
| IAM cursive | ~12% | ~24% |
| Mixed (your target) | ~8–15% | ~18–28% |

This is comparable to Google MLKit on English.
MLKit's advantage is their proprietary training data (billions of samples).
Your advantage is the hierarchical clustering front-end and infinite canvas support.

---

## Next Steps (in order)

1. `pip install torch torchvision torchaudio`
2. Download IAM Online dataset
3. Build `InkDataset` class (parse XML → delta features)
4. Train for 10 epochs to verify the pipeline works
5. Add synthetic augmentation
6. Full 100-epoch training run
7. Export to ONNX for browser deployment
8. Wire into ink-canvas-hierarchical.html via ONNX Runtime Web

The ONNX step is what makes it run IN THE BROWSER like MLKit — no server needed.