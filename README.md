# Black-Face: Skin-Tone–Aware, Deployable Face Recognition for Darker Skin

**CS 153 Final Project — Research track**

A complete, reproducible pipeline that trains a face-recognition model on a
race-balanced corpus, reweights training by **continuous skin tone (ITA)**
instead of coarse race labels, distills it into a smaller model for edge
deployment, and evaluates everything on the **Racial Faces in the Wild (RFW)**
benchmark with a skin-tone–stratified breakdown.

> **Headline result (honest):** under our compute- and data-constrained setting,
> ITA reweighting **did not** improve fairness — it *reduced* African
> verification accuracy (TAR@FAR=1%: 0.390 → 0.331) and *widened* the
> Caucasian–African gap (0.466 → 0.499). We trace this to a concrete confound (a
> truncated dataset download that removed ~half of the African identities) and
> report it as a **cautionary negative result** with direct relevance to
> fairness work in low-resource African data settings. Full analysis in
> [`paper/main.tex`](paper/main.tex).

---

## 1. Problem & motivation

Face recognition is least accurate on African and dark-skinned faces — an equity
*and* deployment problem, since the settings that would most benefit from open,
on-device FR are also the most data- and compute-constrained. Most prior work
balances on **4 race labels**; but a race-*count*-balanced dataset is **not**
skin-tone–balanced (wide skin-tone variation exists *within* each race, and the
darkest-skin tail stays sparse). We test whether reweighting by a continuous,
label-free skin-tone axis — the **Individual Typology Angle (ITA)** — helps the
darkest-skin tail, and whether any gain survives compression to an edge model.

## 2. How it works (architecture)

```
BUPT images ──► prepare_data.sh ──► ImageFolder (flattened, 112×112)
                                         │
                 precompute_ita.py ──────┤  (ITA per image, cached)
                                         ▼
      train.py ──► IR-50 + AdaFace head ─┬─► baseline (uniform sampling)
                                         └─► ITA-reweighted (WeightedRandomSampler)
                                         │
      distill.py ──► IR-50(ITA) ─► IR-18 student (ITA-weighted feature distillation)
                                         │
      evaluate_rfw.py ─► RFW TAR/AUC per race
      evaluate_ita.py ─► TAR per skin-tone (ITA) bin + plot
      results_table.py ─► comparison vs. published baselines
      export_edge.py ─► ONNX + size/latency (verified vs. PyTorch)
```

- **Backbone/head:** AdaFace's IResNet + quality-adaptive margin head, reused
  verbatim (`--adaface-repo`).
- **ITA:** `arctan((L*-50)/b*)` on the centre crop in CIELab; lower = darker.
- **Fairness lever:** `WeightedRandomSampler` with inverse ITA-bin frequency.
- **Distillation:** minimise `1 - cos(student, teacher)` on L2-normed embeddings.

## 3. Repository layout

| Path | What |
|---|---|
| `train.py` | AdaFace/ArcFace/CosFace trainer; `--reweight none|race|ita` |
| `distill.py` | ITA-weighted teacher→student feature distillation |
| `precompute_ita.py` | parallel ITA cache over the training corpus |
| `bupt_labels.py` | BUPT label parsing + inverse-frequency / ITA-bin weights |
| `evaluate_rfw.py` | RFW verification (TAR@FAR, AUC, accuracy) |
| `evaluate_ita.py` | ITA-stratified TAR + the skin-tone plot |
| `results_table.py` | Markdown/LaTeX comparison vs. literature |
| `export_edge.py` | ONNX export + param/size/latency, numerically verified |
| `scripts/prepare_data.sh` | extract + flatten BUPT images, extract RFW |
| `scripts/run_minimal.sh` | one-command end-to-end pipeline (the 6 stages) |
| `tests/` | 18 unit tests (CPU-only, no GPU/data needed) |
| `paper/main.tex` | full writeup (compiles on Overleaf) |
| `results_captured.md` | the exact run numbers |
| `RUNBOOK.md` | cloud (RunPod/A100) provisioning + run guide |

## 4. Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements_train.txt        # torch, torchvision, opencv, onnx, matplotlib, ...
git clone https://github.com/mk-minchul/AdaFace   # backbone + head
export ADAFACE_REPO=$PWD/AdaFace
```
The minimal pipeline uses `--data-format imagefolder` and **does not need mxnet**
(any Python 3.10–3.12). See [`RUNBOOK.md`](RUNBOOK.md) for the exact A100 steps.

## 5. Usage

```bash
# 1. prepare data (extracts BUPT images + RFW; writes data/prepared/paths.env)
bash scripts/prepare_data.sh

# 2. run the whole thing (smoke → ITA cache → baseline → ITA → distill → eval → ONNX)
export ADAFACE_REPO=/path/to/AdaFace
nohup bash scripts/run_minimal.sh > run.log 2>&1 &
tail -f run.log

# outputs land in ./results : results_table.md/.tex, ita_vs_tar.png,
# *_rfw_results.json, ita_binned_results.json, edge_ir_18.onnx
```
Run individual stages directly via `train.py`, `distill.py`, `evaluate_rfw.py`,
etc. — each has `--help`.

```bash
# tests (CPU-only)
python -m pytest -q        # 18 passing
```

## 6. Results

Trained from scratch on BUPT-Balancedface (24,326/≈28,000 identities recovered
from a **truncated** archive; 12 epochs; A100), evaluated on RFW (4×6,000 pairs).

**RFW TAR@FAR=1% by race** (gap = Caucasian − African; lower is fairer):

| Model | African | Asian | Caucasian | Indian | gap |
|---|---|---|---|---|---|
| ArcFace (paper) | 0.840 | 0.922 | 0.942 | 0.903 | 0.102 |
| AdaFace (paper) | 0.941 | 0.962 | 0.974 | 0.951 | 0.033 |
| baseline IR-50 (none) | 0.390 | 0.533 | 0.856 | 0.738 | 0.466 |
| **ITA IR-50** | 0.331 | 0.528 | 0.830 | 0.709 | 0.499 |
| distill IR-18 | 0.281 | 0.516 | 0.784 | 0.656 | 0.504 |

ITA reweighting was **lower in every skin-tone bin**, including the darkest
(0.351 → 0.302). See [`results_captured.md`](results_captured.md) and
[`results_ita_vs_tar.png`](results_ita_vs_tar.png).

**Why (failure analysis):** the truncated download removed ~half of African
identities; up-weighting dark-skin images then concentrates training on a small,
low-diversity African subset → overfitting → worse RFW-African generalisation.
Uniform sampling avoids over-emphasising the depleted group, so it is *fairer*
here. **Lesson: re-balancing toward a group whose own diversity is limited can
amplify, not reduce, disparity** — a real trap for African FR.

## 7. Limitations

- Truncated dataset is the dominant confound; this is **not** a general
  refutation of ITA reweighting, only of its naive use under this data deficit.
- Absolute accuracy is far below fully-trained references (12-epoch matched
  budget + missing data).
- IR-18 (24M params, 96 MB FP32 ONNX) is *smaller*, not mobile-tiny; real edge
  use needs MobileFaceNet + INT8 quantisation.
- The automated CPU-latency benchmark returned an implausible value (host
  contention) and is **not** reported as a result.

## 8. AI-usage disclosure

This project was built with heavy use of an AI coding assistant (**Anthropic
Claude**) for: implementing the training/distillation/eval/export code,
debugging, the multi-hour data transfer and recovery process, and drafting the
README and `paper/main.tex`. **All experiments, design decisions, and the
integrity of the reported numbers are the author's own; no results were
fabricated or altered.** The commit history documents the full iteration,
including the data-recovery saga.

## 9. Citations & acknowledgements

- RFW & BUPT-Balancedface — Mei Wang, Weihong Deng et al., *ICCV 2019* / *CVPR 2020*. Datasets: <http://www.whdeng.cn/RFW/index.html>
- AdaFace — Kim, Jain, Liu, *CVPR 2022*. Code: <https://github.com/mk-minchul/AdaFace>
- ArcFace — Deng et al., *CVPR 2019*.
- ITA — Chardon et al., *Int. J. Cosmetic Science 1991*.
- Knowledge Distillation — Hinton et al., 2015.
- Gender Shades — Buolamwini & Gebru, *FAT\* 2018*.

The backbone (`net.py`) and margin head (`head.py`) are used **from the AdaFace
repository** (cloned at runtime, not vendored); all training/fairness/eval/edge
code in this repo is original.
