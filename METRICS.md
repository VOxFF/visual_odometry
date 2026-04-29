# Trajectory Evaluation Metrics

Dataset: `indoor_forward_7_snapdragon_with_gt`  
Evaluated on 1738 frames (GT window: frames ~815–2608 of 3176 total)  
Divergence threshold: 1.0 m over 5 consecutive frames

---

## Subpixel keypoint sampling

Bilinear depth/flow sampling at float coordinates vs integer nearest-neighbour.

| Metric | Baseline (integer) | Subpixel (bilinear) | Change |
|---|---|---|---|
| ATE RMSE | 11.39 m | **9.10 m** | -20% |
| ATE mean | 10.20 m | **8.29 m** | -19% |
| ATE max | 20.10 m | **17.13 m** | -15% |
| RPE mean (per 10 fr.) | 1.76 m | **1.36 m** | -23% |
| RPE max (per 10 fr.) | 6.37 m | **5.08 m** | -20% |
| % frames within 1.0 m | 8.1% | 5.6% | — |
| Divergence frame | 106 / 1738 | 96 / 1738 | — |

**Conclusion: better.** ~20% reduction across all error metrics. Integer indexing was discarding RAFT's subpixel precision. All subsequent experiments use `subpixel_keypoints: true`.

---

## Keypoint detector: Shi-Tomasi vs uniform grid

Both with `subpixel_keypoints: true`.

| Metric | Uniform grid | Shi-Tomasi corners | Change |
|---|---|---|---|
| ATE RMSE | **9.10 m** | 10.74 m | +18% |
| ATE mean | **8.29 m** | 9.68 m | +17% |
| ATE max | **17.13 m** | 18.99 m | +11% |
| RPE mean (per 10 fr.) | **1.36 m** | 1.71 m | +26% |
| RPE max (per 10 fr.) | **5.08 m** | 6.00 m | +18% |
| % frames within 1.0 m | 5.6% | 6.4% | — |
| Divergence frame | 96 / 1738 | 100 / 1738 | — |

**Conclusion: worse.** In this FPV indoor scene RAFT flow tracks uniform grid points as reliably as detected corners. Subpixel precision matters more than keypoint placement. Reverted to `keypoints_detector: uniform`.

---

## More keypoints: 500 vs 320

Uniform grid, `subpixel_keypoints: true`, `min_depth: 0.0`, `max_keypoints: 500`.

| Metric | Best baseline (320 kp) | 500 kp | Change |
|---|---|---|---|
| ATE RMSE | **9.10 m** | 11.93 m | +31% |
| ATE mean | **8.29 m** | 10.86 m | +31% |
| ATE max | **17.13 m** | 22.83 m | +33% |
| RPE mean (per 10 fr.) | **1.36 m** | 1.62 m | +19% |
| RPE max (per 10 fr.) | **5.08 m** | 6.19 m | +22% |
| % frames within 1.0 m | 5.6% | 3.9% | — |
| Divergence frame | 96 / 1738 | **55 / 1738** | earlier |

**Conclusion: worse.** Extra keypoints land on low-texture regions where uniform grid depth/flow is unreliable — adds noise rather than coverage. Reverted to `max_keypoints: 320`.

---

## dz_threshold sweep — non-deterministic CUDA (Apr 25, unreliable)

`max_keypoints=320`, `subpixel=true`, `uniform`. CUDA non-deterministic (no `cudnn.deterministic`).

| Label | dz_threshold | log_dz | ATE RMSE | Div frame |
|---|---|---|---|---|
| dz=0.5, log=F | 0.5 | no | 8.13 m | 132 |
| dz=1.0, log=T | 1.0 | yes | 10.06 m | 100 |
| dz=2.0, log=T | 2.0 | yes | 10.22 m | 105 |
| dz=0.5, log=T | 0.5 | yes | 11.21 m | 75 |
| dz=1.0, log=F | 1.0 | no | 11.84 m | 107 |
| dz=2.0, log=F | 2.0 | no | 13.88 m | 94 |

**These results are misleading** — CUDA non-determinism swamped the config signal. dz=0.5 ran first (fresh GPU, lucky) and dz=1.0 ran fifth (unfavorable GPU state). Do not use for conclusions.

---

## dz_threshold sweep — deterministic CUDA (Apr 27, reliable)

Same grid. Added `torch.backends.cudnn.deterministic=True` + `benchmark=False` to both RAFT solvers.

| Label | dz_threshold | log_dz | ATE RMSE | ATE mean | RPE mean | Div frame |
|---|---|---|---|---|---|---|
| **dz=1.0, log=F** | 1.0 | no | **8.62 m** | **7.68 m** | **1.31 m** | 101 |
| dz=2.0, log=F | 2.0 | no | 8.96 m | 7.97 m | 1.41 m | 105 |
| dz=2.0, log=T | 2.0 | yes | 9.68 m | 8.61 m | 1.41 m | **121** |
| dz=1.0, log=T | 1.0 | yes | 11.69 m | 10.28 m | 1.79 m | 84 |
| dz=0.5, log=F | 0.5 | no | 11.98 m | 10.64 m | 1.70 m | 109 |
| dz=0.5, log=T | 0.5 | yes | 13.49 m | 12.05 m | 2.08 m | 80 |

**Conclusion: `dz_threshold=1.0, log=False` is the actual best.** Tighter filter (0.5) hurts — removes too many valid keypoints. Looser (2.0) is close. `log_dz` consistently hurts. Deterministic mode itself improved results vs prior standalone runs (8.62m vs 9.10m).

---

## min_depth filter + more keypoints

Uniform grid, `subpixel_keypoints: true`, `min_depth: 0.3`, `max_keypoints: 500`.

| Metric | Best baseline | min_depth=0.3, 500 kp | Change |
|---|---|---|---|
| ATE RMSE | **9.10 m** | 13.44 m | +48% |
| ATE mean | **8.29 m** | 12.01 m | +45% |
| ATE max | **17.13 m** | 23.38 m | +37% |
| RPE mean (per 10 fr.) | **1.36 m** | 1.87 m | +38% |
| RPE max (per 10 fr.) | **5.08 m** | 6.82 m | +34% |
| % frames within 1.0 m | 5.6% | 5.4% | — |
| Divergence frame | 96 / 1738 | 94 / 1738 | — |

**Conclusion: worse.** `min_depth=0.3` filters too many valid keypoints, leaving SVD under-constrained. Extra keypoints (500) did not compensate. Reverted to `min_depth: 0.0`, `max_keypoints: 320`.

---

## History

All files in `run_v1/` output directory:

| File | Config | ATE RMSE |
|---|---|---|
| `trajectory_eval_20260422_215411.txt` | Baseline: integer sampling, uniform, 320 kp | 11.39 m |
| `trajectory_eval_20260422_220303.txt` | Baseline duplicate | 11.39 m |
| `trajectory_eval_20260424_204240 (subpixel).txt` | **Subpixel + uniform, 320 kp** — best so far | **9.10 m** |
| `trajectory_eval_20260424_211716.txt` | Shi-Tomasi (unlabeled run) | 10.74 m |
| `trajectory_eval_20260424_211847 (tomasi).txt` | Shi-Tomasi + subpixel | 10.74 m |
| `trajectory_eval_20260424_214814.txt` | Shi-Tomasi duplicate | 10.74 m |
| `trajectory_eval_20260425_145018.txt` | Uniform + subpixel, 500 kp, min_depth=0.3 — reverted | 13.44 m |
| `trajectory_eval_20260425_152503.txt` | Uniform + subpixel, 500 kp, min_depth=0.0 — reverted | 11.93 m |
| `trajectory_eval_20260425_154941.txt` | Uniform + subpixel, 320 kp — repeat run (GPU non-determinism) | 9.45 m |
| sweep `20260427_234144.csv` | dz sweep with deterministic CUDA — best: dz=1.0 log=F | **8.62 m** |

---

## Notes
- Pre-alignment drift (~7.2 m) is stable across all configs — accumulated error in the first ~815 frames before GT coverage begins, unaffected by keypoint strategy
- Divergence consistently occurs around frame 96–106, corresponding to the first drastic camera swing
- Best config to date: `subpixel_keypoints: true`, `keypoints_detector: uniform`, `max_keypoints: 320`, `min_depth: 0.0`, `dz_threshold: 1.0`, `log_dz_threshold: false` → **ATE RMSE 8.62 m**
- CUDA non-determinism was masking real config signal — added `cudnn.deterministic=True` to both RAFT solvers; results now reproducible
- `dz_threshold=0.5` appeared best in non-deterministic sweep but is actually 5th worst — was a lucky first-run artifact
- Raising `min_depth` to 0.3 m hurt badly (+48%) — filters too many valid keypoints
- Raising `max_keypoints` to 500 hurt (+31%, divergence at frame 55) — extra points land on low-texture regions
- `log_dz_threshold` consistently hurts across all dz values
