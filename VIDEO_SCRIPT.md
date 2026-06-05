# Demo Video Script (~6 min) — Black-Face

Target 5–7 min (10 is the cap). Screen-record: README → a code/script walkthrough
→ the results table & plot → `pytest` passing → `paper/main.tex`. Speak to the
four required questions. **Lean into honesty — the rubric explicitly rewards
failure analysis, limitations, and disclosure.**

---

## 0:00 – 0:40 — Hook & what it is
> "Face recognition is least accurate on African and dark-skinned faces — and
> the places that would benefit most from open, on-device face ID are exactly
> the ones the tech serves worst. I built **Black-Face**: an end-to-end pipeline
> that trains a face-recognition model to be fairer on darker skin using a
> *continuous skin-tone* signal instead of coarse race labels, distills it for
> edge devices, and rigorously measures the result on a standard bias benchmark.
> I'll show what I built, what I found — including an honest negative result —
> and why that finding is itself useful."

## 0:40 – 2:00 — Q1: Why I built this (problem & insight)
- Show README §1. Bottlenecks identified:
  - Demographic bias in FR is well documented (cite Gender Shades, RFW).
  - **Insight:** prior bias work balances on *4 race labels*, but a
    race-count-balanced dataset is **not skin-tone-balanced** — the darkest-skin
    tail is sparse *within* every race.
  - So I use the **Individual Typology Angle (ITA)**, a continuous, label-free
    skin-tone measure, as the fairness lever — and ask whether it helps the
    darkest tail and whether gains survive compression to an edge model.

## 2:00 – 4:00 — Q2: How it works (research track)
Walk the architecture diagram in README §2, then show code briefly:
- **Data:** BUPT-Balancedface (~28K identities, 4 races). `prepare_data.sh`
  extracts + flattens to an ImageFolder.
- **Skin tone:** `precompute_ita.py` computes ITA per image (show the formula).
- **Training:** `train.py` — AdaFace IR-50; the fairness lever is a
  `WeightedRandomSampler` that up-samples low-ITA (darker) images
  (`--reweight ita`) vs. a uniform baseline.
- **Distillation:** `distill.py` compresses IR-50 → IR-18 (edge).
- **Evaluation:** `evaluate_rfw.py` + `evaluate_ita.py` → per-race **and**
  per-skin-tone TAR, plus `export_edge.py` to ONNX.
- Show `pytest -q` → **18 passing**, and the one-command `run_minimal.sh`.

## 4:00 – 5:30 — Q3 + Evidence: what I found & impact
- Show the **results table** (README §6) and **`ita_vs_tar.png`**.
- State it plainly:
  > "My hypothesis was that ITA reweighting would lift the darkest-skin tail. It
  > did the **opposite** — African accuracy dropped and the racial gap widened."
- Then the **failure analysis** (this is the strongest part for grading):
  > "The cause is concrete: my dataset download was *truncated* and lost about
  > half the African identities. Up-weighting dark-skin images then over-trains
  > on a tiny, low-diversity African subset — overfitting. The lesson
  > generalizes: **re-balancing toward a group whose own data diversity is
  > limited can amplify, not reduce, disparity** — exactly the trap to avoid in
  > low-resource African FR."
- **Impact / use cases:** an open, reproducible pipeline + benchmark harness for
  skin-tone-fair FR; a cautionary finding for anyone doing fairness work on
  scarce demographic data; a base others can run correctly once data is complete.

## 5:30 – 6:00 — Q4: What I'd add next + integrity
- Future work: remove the confound (checksum-verified full dataset, full
  training schedule); ITA-tail validation + early stopping; MobileFaceNet + INT8
  for true edge with honest on-device latency; ITA-aware *loss* terms instead of
  sampling alone.
- Close on integrity:
  > "Everything is in the public repo with full commit history — including the
  > messy data-recovery process. I used Claude as an AI coding assistant
  > throughout, disclosed in the README; all results are real and unaltered."

---

### Recording checklist
- [ ] README visible and scrolled through (overview, results, AI disclosure, citations)
- [ ] One code file shown (`train.py` sampler or `precompute_ita.py` ITA)
- [ ] `python -m pytest -q` → 18 passed, on screen
- [ ] Results table + `ita_vs_tar.png` on screen
- [ ] `paper/main.tex` (or compiled PDF) shown briefly
- [ ] You state the negative result AND the failure analysis out loud
- [ ] You state AI usage out loud
- [ ] Repo is **public / shared with course staff** before submitting
```
