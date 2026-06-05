# RUNBOOK — minimal fair-edge face-recognition run (A100, <$25, <24h)

The whole pipeline is wired and verified end-to-end on CPU with a tiny fake
dataset. This is the exact sequence to run it for real on a single A100. The
minimal path uses `--data-format imagefolder`, so **you do not need mxnet** and
any Python 3.10–3.12 works.

What you produce: a baseline IR-50, an ITA-reweighted IR-50 (the headline fair
model), an ITA-weighted distilled `ir_18` edge model, RFW metrics + an
ITA-stratified skin-tone breakdown + the comparison table + an ONNX export with
CPU latency.

---

## 0. Provision the pod
- **RunPod** (recommended for reliability) or Vast.ai.
- GPU: **1× A100 80GB**. Template: a PyTorch/CUDA image (e.g. "RunPod PyTorch").
- **Persistent volume / network volume: ~100 GB**, mounted at `/workspace`. Put
  data + code + outputs here so a pod restart doesn't lose anything. (The images
  archive is ~27 GB and extracts to ~27 GB; delete the archive after a successful
  extract to reclaim space — see step 4.)
- Note the SSH command from the dashboard.

## 1. Get the code on the pod
```bash
cd /workspace
git clone https://github.com/Laoluog/blackfac3 black-face
cd black-face
```

## 2. Get the data on the pod
You need two things in `/workspace/black-face/data/`:
- the **BUPT-Balancedface images** archive — `Equalizedface.tar.gz` (~27 GB, the
  one whose contents are `race_per_7000/<Race>/<identity>/*.jpg`). Note this is a
  *different* file from `Equalizedface (1).tar.gz`, which is the mxnet `.rec` and
  is **not** used by this pipeline.
- **`test.tar.gz`** (RFW test set).

Fastest is to download them *directly on the pod* (datacenter bandwidth beats
your home upload). If they only live on your laptop, push them up, e.g.:
```bash
# from your laptop (RunPod gives you the host/port):
rsync -avP "data/test.tar.gz" "data/<BUPT-images-archive>" \
      root@<POD_HOST>:/workspace/black-face/data/
```
(`runpodctl send` / Vast's web upload also work.)

## 3. Install deps + AdaFace backbone
```bash
cd /workspace/black-face
pip install -r requirements_train.txt          # torch is already on the image; this adds the rest
git clone https://github.com/mk-minchul/AdaFace /workspace/AdaFace
export ADAFACE_REPO=/workspace/AdaFace
python -c "import sys; sys.path.insert(0,'$ADAFACE_REPO'); import net, head; print('AdaFace OK')"
python -c "import torch; print('cuda', torch.cuda.is_available())"   # must print True
```

## 4. Prepare data (extract + auto-detect paths)
```bash
bash scripts/prepare_data.sh
# Extracts test.tar.gz -> RFW; extracts the images archive -> data/prepared/BUPT-raw
# (race_per_7000/<Race>/<identity>/...), then FLATTENS it into data/prepared/BUPT-flat
# as one symlink per identity so ImageFolder sees ~28000 identity classes (not the
# 4 races). Writes data/prepared/paths.env with IMAGES_ROOT + RFW_ROOT.
#
# If auto-detect picks the wrong archive, point it explicitly:
#   BUPT_IMAGES_ARCHIVE=/workspace/black-face/data/Equalizedface.tar.gz \
#   bash scripts/prepare_data.sh
```
Confirm it prints `IMAGES_ROOT` with **~28000 identity dirs** and a valid `RFW_ROOT`.
Then reclaim space (optional but recommended on a 100 GB volume):
```bash
rm data/Equalizedface.tar.gz        # the archive; extracted copy lives in BUPT-raw
```

## 5. Launch the run (backgrounded, survives logout)
```bash
export ADAFACE_REPO=/workspace/AdaFace
nohup bash scripts/run_minimal.sh > run.log 2>&1 &
tail -f run.log
```
It runs in order: smoke test → ITA precompute → IR-50 baseline → IR-50+ITA →
distill→ir_18 → eval all → ONNX export. **Each step is resumable** — if the pod
dies, just re-run the same command and it skips finished steps.

**Knobs** (env vars before the `nohup`):
- `EPOCHS=12` (default; bump to `26` for max quality if you have wall-clock to spare)
- `BATCH=512` (A100 80GB; lower to 256 if you see OOM)
- `STUDENT_ARCH=ir_18` (the edge model)

Rough A100 wall-clock at `EPOCHS=12`: ITA precompute ~30–60 min (one-time),
each IR-50 run ~3.5–4 h, distill ~2.5 h, eval ~1 h → **~12–15 h total, well under
your 24 h and ~$10–18.**

## 6. When it finishes
```bash
cat results/run.log 2>/dev/null; cat results/edge_summary.txt
ls results/
```
**Send me back this whole `results/` folder + `run.log`:**
- `results_table.md` / `results_table.tex` — RFW vs literature baselines
- `*_rfw_results.json` — per-model RFW metrics (baseline / ITA / distilled)
- `ita_binned_results.json` — per-skin-tone-bin TAR (the fairness evidence)
- `ita_vs_tar.png` — the money plot
- `edge_summary.txt` + `edge_ir_18.onnx` — size / params / CPU latency
- `run.log` — training curves

Easiest: `tar czf results.tgz results run.log` then `rsync` it down (or
`runpodctl send`). With those I'll build the paper's tables, figures, and
results/methods narrative.

## 7. STOP BILLING
**Destroy/terminate the pod** as soon as `results.tgz` is on your laptop. If you
used a persistent volume and want to keep checkpoints, download
`experiments/*_last.ckpt` first (the IR-50+ITA and ir_18 student are the keepers).

---

### Honesty guardrails (matters for the paper)
- Train **from scratch on BUPT** (this pipeline does). Do **not** fine-tune a
  pretrained AdaFace model — those are trained on MS1M/WebFace, and RFW is drawn
  from MS-Celeb-1M, so that eval would be contaminated (the RFW readme says so).
- Fewer epochs lowers absolute accuracy; frame it as a **matched-budget controlled
  comparison**. The *delta* between baseline and ITA (and its survival through
  distillation) is the contribution and is valid regardless of absolute numbers.
- Report real numbers. The story (small + fair, bias preserved under compression)
  stands on an honest delta; it does not need inflation.
