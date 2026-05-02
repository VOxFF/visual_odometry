"""
Measure RAFT-Stereo disparity/depth noise on stationary frames.

Method: camera is stationary for frames 0-883. Running RAFT on consecutive
frames of the same static scene gives pure measurement noise (no true motion).
We compute per-pixel temporal std of disparity across N frames, then bin by
depth to compare against the theoretical model:

    σ_z = z² / (fx * baseline) * σ_disp_effective
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def _setup_external_paths(base_dir):
    raft_stereo_path = os.path.join(base_dir, "external", "RAFT-Stereo")
    raft_flow_path   = os.path.join(base_dir, "external", "RAFT-Flow")
    core_path        = os.path.join(raft_flow_path, "flow_core")
    for p in [raft_stereo_path, raft_flow_path]:
        if p not in sys.path:
            sys.path.append(p)
    if core_path not in sys.path:
        sys.path.insert(0, core_path)

_setup_external_paths(project_root)

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

from config.config import Config
from modules.stereo.stereo_depth import StereoDepth
from modules.stereo.stereo_params_YAML import StereoParamsYAML
from modules.stereo.stereo_rectification import StereoRectification
from modules.stereo.stereo_disparity_RAFT import DisparityRAFT

# Frames to sample from the stationary phase (camera not moving)
# Avoiding the very start and end of stationary phase to skip any handling
SAMPLE_START = 20
SAMPLE_COUNT = 20   # number of consecutive frames to run RAFT on


def load_stereo(dataset_path, frame_idx):
    left  = os.path.join(dataset_path, f"img/image_0_{frame_idx}.png")
    right = os.path.join(dataset_path, f"img/image_1_{frame_idx}.png")
    l = cv2.imread(left,  cv2.IMREAD_GRAYSCALE)
    r = cv2.imread(right, cv2.IMREAD_GRAYSCALE)
    return l, r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--start", type=int, default=SAMPLE_START,
                        help="First frame index to sample (default: %(default)s)")
    parser.add_argument("--count", type=int, default=SAMPLE_COUNT,
                        help="Number of consecutive frames (default: %(default)s)")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    stereo_ckpt = cfg.stereo_checkpoint if os.path.isabs(cfg.stereo_checkpoint) \
                  else os.path.join(project_root, cfg.stereo_checkpoint)

    params        = StereoParamsYAML(cfg.yaml_file)
    rectification = StereoRectification(params)
    disp_solver   = DisparityRAFT(stereo_ckpt, rectification, cfg.raft_iters, False)
    depth_solver  = StereoDepth(params)

    stereo_mask, _, _, _ = rectification.get_rectification_masks()
    K   = params.get_camera_params(
            __import__('modules.stereo.stereo_interfaces', fromlist=['StereoParamsInterface'])
            .StereoParamsInterface.StereoCamera.LEFT).get_intrinsics()
    fx       = float(K[0, 0])
    fy       = float(K[1, 1])
    baseline = float(params.get_baseline())

    sample_start = args.start
    sample_count = args.count

    print(f"fx={fx:.1f}px  baseline={baseline*100:.1f}cm")
    print(f"Sampling frames {sample_start} – {sample_start + sample_count - 1}")

    # ── Collect N disparity maps ───────────────────────────────────────────────
    disps, depths = [], []
    for idx in range(sample_start, sample_start + sample_count):
        l, r = load_stereo(cfg.dataset_path, idx)
        if l is None or r is None:
            print(f"  frame {idx}: missing"); continue
        d = disp_solver.compute_disparity(l, r)
        z = depth_solver.compute_depth(d)
        disps.append(d)
        depths.append(z)
        print(f"  frame {idx}: disp range [{d[stereo_mask].min():.2f}, {d[stereo_mask].max():.2f}]px  "
              f"depth range [{z[stereo_mask & (z>0)].min():.2f}, {z[stereo_mask & (z>0)].max():.2f}]m")

    disps  = np.stack(disps,  axis=0)   # (N, H, W)
    depths = np.stack(depths, axis=0)   # (N, H, W)
    mask2d = stereo_mask & (depths.mean(axis=0) > 0) & (depths.mean(axis=0) < cfg.max_depth)

    # ── Per-pixel temporal statistics ─────────────────────────────────────────
    disp_std  = disps.std(axis=0)    # (H, W)  disparity std over time
    depth_std = depths.std(axis=0)   # (H, W)  depth std over time
    depth_mean = depths.mean(axis=0) # (H, W)

    valid = mask2d & (disp_std > 0)

    print(f"\n── Disparity noise (temporal std over {SAMPLE_COUNT} frames) ──")
    print(f"  mean σ_disp = {disp_std[valid].mean():.3f} px")
    print(f"  median σ_disp = {np.median(disp_std[valid]):.3f} px")
    print(f"  p95 σ_disp = {np.percentile(disp_std[valid], 95):.3f} px")
    print(f"  max σ_disp = {disp_std[valid].max():.3f} px")

    print(f"\n── Depth noise (temporal std over {SAMPLE_COUNT} frames) ──")
    print(f"  mean σ_z = {depth_std[valid].mean():.3f} m")
    print(f"  median σ_z = {np.median(depth_std[valid]):.3f} m")
    print(f"  p95 σ_z = {np.percentile(depth_std[valid], 95):.3f} m")

    # ── Noise vs depth: measured vs theoretical ────────────────────────────────
    print(f"\n── σ_z vs depth: measured vs theoretical (σ_disp_eff from data) ──")
    depth_bins = [0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 15]
    sigma_disp_eff = float(np.median(disp_std[valid]))  # use measured as effective σ_disp

    print(f"{'depth':>8}  {'n_px':>8}  {'σ_z meas':>10}  {'σ_z theory':>12}  {'ratio':>8}")
    print("-" * 55)
    for lo, hi in zip(depth_bins[:-1], depth_bins[1:]):
        bin_mask = valid & (depth_mean >= lo) & (depth_mean < hi)
        if bin_mask.sum() < 10:
            continue
        sz_meas   = depth_std[bin_mask].mean()
        z_mid     = depth_mean[bin_mask].mean()
        sz_theory = z_mid**2 / (fx * baseline) * sigma_disp_eff
        print(f"  {lo:.1f}-{hi:.1f}m  {bin_mask.sum():>8d}  "
              f"{sz_meas:>10.3f}m  {sz_theory:>12.3f}m  {sz_meas/sz_theory:>8.2f}x")

    # ── Back out effective σ_disp per depth bin ────────────────────────────────
    print(f"\n── Effective σ_disp implied by measured depth noise ──")
    for lo, hi in zip(depth_bins[:-1], depth_bins[1:]):
        bin_mask = valid & (depth_mean >= lo) & (depth_mean < hi)
        if bin_mask.sum() < 10:
            continue
        sz_meas  = depth_std[bin_mask].mean()
        z_mid    = depth_mean[bin_mask].mean()
        # σ_disp = σ_z * fx * baseline / z²
        sd_eff   = sz_meas * fx * baseline / (z_mid**2)
        print(f"  {lo:.1f}-{hi:.1f}m: σ_disp_eff = {sd_eff:.3f} px")

    # ── Spatial noise: std of disparity within 5x5 patches on single frame ────
    ref_disp = disps[len(disps)//2]
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(ref_disp.astype(np.float32), size=5)
    local_sq   = uniform_filter(ref_disp.astype(np.float32)**2, size=5)
    spatial_std = np.sqrt(np.maximum(local_sq - local_mean**2, 0))
    print(f"\n── Spatial smoothness (5×5 patch std on single frame) ──")
    print(f"  mean patch σ_disp = {spatial_std[mask2d].mean():.3f} px")
    print(f"  median             = {np.median(spatial_std[mask2d]):.3f} px")

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    # Mean depth map
    ax = fig.add_subplot(gs[0, 0])
    dm = depth_mean.copy(); dm[~mask2d] = 0
    im = ax.imshow(dm, cmap='plasma', vmin=0, vmax=cfg.max_depth)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Mean depth (m)")

    # Disparity std map
    ax = fig.add_subplot(gs[0, 1])
    ds = disp_std.copy(); ds[~mask2d] = 0
    im = ax.imshow(ds, cmap='hot', vmin=0, vmax=np.percentile(disp_std[valid], 99))
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"Temporal σ_disp (px)  —  {SAMPLE_COUNT} frames")

    # Depth std map
    ax = fig.add_subplot(gs[0, 2])
    zs = depth_std.copy(); zs[~mask2d] = 0
    im = ax.imshow(zs, cmap='hot', vmin=0, vmax=np.percentile(depth_std[valid], 99))
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Temporal σ_z (m)")

    # σ_disp histogram
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(disp_std[valid].ravel(), bins=80, color='steelblue', edgecolor='none')
    ax.axvline(np.median(disp_std[valid]), color='red', linestyle='--',
               label=f"median={np.median(disp_std[valid]):.3f}px")
    ax.set_xlabel("σ_disp (px)"); ax.set_ylabel("pixels"); ax.set_title("Disparity noise distribution")
    ax.legend()

    # σ_z vs depth scatter
    ax = fig.add_subplot(gs[1, 1])
    sample_mask = valid & (np.random.rand(*valid.shape) < 0.05)  # 5% sample
    ax.scatter(depth_mean[sample_mask], depth_std[sample_mask],
               s=1, alpha=0.3, color='steelblue', label='measured')
    zz = np.linspace(0.5, cfg.max_depth, 100)
    for sd, ls, lbl in [(0.25, '--', 'theory σ_d=0.25px'),
                         (0.5,  '-',  'theory σ_d=0.5px'),
                         (1.0,  ':',  'theory σ_d=1.0px')]:
        ax.plot(zz, zz**2 / (fx * baseline) * sd, ls, label=lbl)
    ax.set_xlabel("depth (m)"); ax.set_ylabel("σ_z (m)")
    ax.set_title("Depth noise vs depth"); ax.legend(fontsize=7); ax.set_ylim(0, 3)

    # Spatial std map
    ax = fig.add_subplot(gs[1, 2])
    ss = spatial_std.copy(); ss[~mask2d] = 0
    im = ax.imshow(ss, cmap='hot', vmin=0, vmax=np.percentile(spatial_std[mask2d], 99))
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Spatial σ_disp (5×5 patch, px)")

    plt.suptitle(f"RAFT-Stereo depth noise  —  baseline={baseline*100:.1f}cm  fx={fx:.0f}px",
                 fontsize=13)
    plt.tight_layout()

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(cfg.output_path,
                            f"depth_noise_f{sample_start}-{sample_start+sample_count-1}_{ts}.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
