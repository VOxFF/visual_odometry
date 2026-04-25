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

---

## Notes
- Pre-alignment drift (~7.2 m) is stable across all configs — accumulated error in the first ~815 frames before GT coverage begins, unaffected by keypoint strategy
- Divergence consistently occurs around frame 96–106, corresponding to the first drastic camera swing
- Best config to date: `subpixel_keypoints: true`, `keypoints_detector: uniform`, `max_keypoints: 320`, `min_depth: 0.0`
- Raising `min_depth` to 0.3 m hurt badly (+48% ATE RMSE) — filters too many valid keypoints, SVD becomes under-constrained
