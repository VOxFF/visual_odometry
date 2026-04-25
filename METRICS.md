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

## Notes
- Pre-alignment drift (~7.2 m) is stable across all configs — accumulated error in the first ~815 frames before GT coverage begins, unaffected by keypoint strategy
- Divergence consistently occurs around frame 96–106, corresponding to the first drastic camera swing
- Best config to date: `subpixel_keypoints: true`, `keypoints_detector: uniform`
