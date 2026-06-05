# Captured results (from run.log, 2026-06-04)

Pipeline: from-scratch IR-50 on BUPT-Balancedface images (TRUNCATED download:
24,326 of ~28,000 identities, 887,331 images; African race ~half missing).
12 epochs, batch 512, A100. Evaluated on RFW (4×6000 pairs).

## RFW per-race (full) — TAR@FAR

### ita_ir_50 (reweight=ita, headline)
| Race | AUC | Acc | TAR@0.1% | TAR@1% | TAR@10% |
|---|---|---|---|---|---|
| African   | 0.8620 | 0.7855 | 0.1540 | 0.3307 | 0.6447 |
| Asian     | 0.9132 | 0.8403 | 0.2623 | 0.5280 | 0.7777 |
| Caucasian | 0.9807 | 0.9357 | 0.7027 | 0.8300 | 0.9497 |
| Indian    | 0.9578 | 0.9007 | 0.5800 | 0.7093 | 0.8933 |

### distill_ir_18 (ITA-weighted distillation of ita_ir_50)
| Race | AUC | Acc | TAR@0.1% | TAR@1% | TAR@10% |
|---|---|---|---|---|---|
| African   | 0.8338 | 0.7650 | 0.1523 | 0.2807 | 0.5957 |
| Asian     | 0.9015 | 0.8277 | 0.2863 | 0.5157 | 0.7400 |
| Caucasian | 0.9729 | 0.9215 | 0.6543 | 0.7843 | 0.9250 |
| Indian    | 0.9456 | 0.8877 | 0.5390 | 0.6557 | 0.8673 |

### baseline_ir_50 (reweight=none) — only TAR@1% captured
African 0.3897 · Asian 0.5330 · Caucasian 0.8560 · Indian 0.7377
(AUC / Acc / TAR@0.1% / TAR@10% were not in the pasted log; on the network
volume if it still exists, otherwise re-derivable by re-running eval.)

## RFW TAR@1% comparison table (vs literature)
| Model | African | Asian | Caucasian | Indian | Cauc−African gap |
|---|---|---|---|---|---|
| ArcFace (paper)  | 0.8398 | 0.9218 | 0.9415 | 0.9028 | 0.1017 |
| AdaFace (paper)  | 0.9410 | 0.9620 | 0.9740 | 0.9510 | 0.0330 |
| baseline_ir_50   | 0.3897 | 0.5330 | 0.8560 | 0.7377 | 0.4663 |
| ita_ir_50        | 0.3307 | 0.5280 | 0.8300 | 0.7093 | 0.4993 |
| distill_ir_18    | 0.2807 | 0.5157 | 0.7843 | 0.6557 | 0.5037 |

## ITA-binned TAR@FAR=0.01 (5 equal-population skin-tone bins)
bin means (deg): b0 -63.35 [-90,-48.1] (darkest) · b1 -35.69 · b2 -12.48 · b3 8.47 · b4 41.66 [20,90] (lightest)
| Model | b0 | b1 | b2 | b3 | b4 |
|---|---|---|---|---|---|
| baseline_ir_50 | 0.3514 | 0.4180 | 0.5636 | 0.5532 | 0.5469 |
| ita_ir_50      | 0.3018 | 0.3730 | 0.5139 | 0.4874 | 0.5202 |
| distill_ir_18  | 0.2734 | 0.2923 | 0.4558 | 0.5031 | 0.4605 |

## Edge export (ir_18)
params 24.02 M · onnx 96.11 MB · onnx-vs-torch max abs diff 2.27e-07 (export verified)
cpu_latency 17114 ms/image — MEASUREMENT ARTIFACT (pod CPU contention/threads); DO NOT report; re-benchmark on target hardware.

## Headline finding (honest)
ITA reweighting REDUCED African TAR@1% (0.390 → 0.331), reduced the darkest ITA
bin (0.351 → 0.302), and WIDENED the Cauc−African gap (0.466 → 0.499).
Most likely confound: the truncated download removed ~half the African identities,
so up-sampling dark-skin images concentrated training on a depleted, low-diversity
African set → overfitting → worse RFW-African generalization. 12 epochs is also short.
Conclusion: a *negative/cautionary* result, confounded by data scarcity — not a clean refutation of ITA reweighting.
