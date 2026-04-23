
import os
import sys
import re
import ast
import argparse
import itertools
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config import Config
from modules.imu.imu_YAML import YamlIMU
from modules.io.data_utils import match_ground_truth_positions

# ── Script-level knobs ─────────────────────────────────────────────────────────
divergence_threshold = 1.0   # metres — frame is "diverged" beyond this error
rpe_segment_len      = 10    # frames per RPE segment
# ──────────────────────────────────────────────────────────────────────────────


def load_estimated_trajectory(traj_path, limit=0):
    """Reconstruct global positions from camera_trajectory.txt relative poses."""
    T_global = np.eye(4)
    positions = []

    with open(traj_path, 'r') as f:
        f.readline()  # skip header
        lines = itertools.islice(f, limit) if limit else f
        for line in lines:
            matches = re.findall(r'\[.*?\]', line)
            if len(matches) < 2:
                continue
            t_rel = np.array(ast.literal_eval(matches[0]))
            R_rel = np.array(ast.literal_eval(matches[1])).reshape(3, 3)

            T_rel_inv = np.eye(4)
            T_rel_inv[:3, :3] = R_rel.T
            T_rel_inv[:3, 3]  = -R_rel.T @ t_rel
            T_global = T_global @ T_rel_inv

            positions.append(T_global[:3, 3].copy())

    return positions


def compute_ate(est_positions, gt_positions):
    """
    Absolute Trajectory Error (ATE) — origin-aligned, no scale correction.
    Returns per-frame errors and RMSE.
    """
    est = np.array(est_positions)
    gt  = np.array(gt_positions)

    # Origin-align both to first frame
    est = est - est[0]
    gt  = gt  - gt[0]

    errors = np.linalg.norm(est - gt, axis=1)
    return errors, float(np.sqrt(np.mean(errors ** 2)))


def compute_rpe(est_positions, gt_positions, segment_len):
    """
    Relative Pose Error (RPE) — translation error over fixed-length segments.
    Returns per-segment translation errors (metres).
    """
    est = np.array(est_positions)
    gt  = np.array(gt_positions)

    errors = []
    for i in range(0, len(est) - segment_len, segment_len):
        delta_est = est[i + segment_len] - est[i]
        delta_gt  = gt[i + segment_len]  - gt[i]
        errors.append(float(np.linalg.norm(delta_est - delta_gt)))

    return np.array(errors)


def find_divergence_frame(per_frame_errors, threshold):
    """First frame where error exceeds threshold and stays above it for >5 consecutive frames."""
    above = per_frame_errors > threshold
    for i in range(len(above) - 5):
        if above[i:i+5].all():
            return i
    return None


def main():
    parser = argparse.ArgumentParser(description="Trajectory evaluation metrics")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    traj_path  = os.path.join(cfg.output_path, "camera_trajectory.txt")
    gt_path    = os.path.join(cfg.dataset_path, "groundtruth.txt")
    left_txt   = os.path.join(cfg.dataset_path, "left_images.txt")

    if not os.path.exists(traj_path):
        print(f"Trajectory file not found: {traj_path}")
        return

    print("Loading estimated trajectory...")
    est_positions = load_estimated_trajectory(traj_path, cfg.limit)

    print("Loading initial pose from IMU...")
    imu = YamlIMU(cfg.yaml_file, left_txt, gt_path)
    R_cam_in_world = imu.get_initial_pose()

    # Rotate estimated positions into world frame
    est_world = [R_cam_in_world @ p for p in est_positions]

    print("Matching with ground truth...")
    combined = match_ground_truth_positions(est_world, left_txt, gt_path)
    valid = [(est, gt) for est, gt in combined if gt is not None]

    if not valid:
        print("No GT matches found.")
        return

    est_matched = [p for p, _ in valid]
    gt_matched  = [p for _, p in valid]

    # Pre-alignment drift: how far was the estimate from GT at the first matched frame
    # (before origin-alignment hides it)
    pre_alignment_drift = float(np.linalg.norm(
        np.array(est_matched[0]) - np.array(gt_matched[0])
    ))

    # ── Metrics ───────────────────────────────────────────────────────────────
    per_frame_errors, ate_rmse = compute_ate(est_matched, gt_matched)
    rpe_errors = compute_rpe(est_matched, gt_matched, rpe_segment_len)
    div_frame  = find_divergence_frame(per_frame_errors, divergence_threshold)

    pct_good = float(np.mean(per_frame_errors < divergence_threshold) * 100)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    div_line = (f"  Divergence at frame   : {div_frame} / {len(per_frame_errors)}"
                if div_frame is not None
                else f"  Divergence at frame   : never exceeded {divergence_threshold:.1f}m threshold")

    report = (
        f"── Trajectory Evaluation  [{ts}] ────────────────────\n"
        f"  Frames evaluated      : {len(per_frame_errors)}\n"
        f"  Pre-alignment drift   : {pre_alignment_drift:.4f} m  (accumulated before GT window)\n"
        f"  ATE RMSE              : {ate_rmse:.4f} m\n"
        f"  ATE max               : {per_frame_errors.max():.4f} m\n"
        f"  ATE mean              : {per_frame_errors.mean():.4f} m\n"
        f"  RPE mean (per {rpe_segment_len:>3} fr.) : {rpe_errors.mean():.4f} m\n"
        f"  RPE max  (per {rpe_segment_len:>3} fr.) : {rpe_errors.max():.4f} m\n"
        f"  % frames within {divergence_threshold:.1f}m  : {pct_good:.1f}%\n"
        f"{div_line}\n"
        f"──────────────────────────────────────────────────────\n"
    )

    print(f"\n{report}")

    txt_path = os.path.join(cfg.output_path, f"trajectory_eval_{ts}.txt")
    with open(txt_path, 'w') as f:
        f.write(report)
    print(f"Metrics saved to: {txt_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Per-frame ATE
    axes[0].plot(per_frame_errors, linewidth=0.8, color='steelblue', label='Per-frame error (m)')
    axes[0].axhline(divergence_threshold, color='red', linestyle='--', linewidth=1, label=f'Divergence threshold ({divergence_threshold}m)')
    if div_frame is not None:
        axes[0].axvline(div_frame, color='orange', linestyle='--', linewidth=1, label=f'Divergence frame {div_frame}')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Position error (m)')
    axes[0].set_title(f'Per-frame ATE  |  RMSE: {ate_rmse:.4f}m  |  {pct_good:.1f}% within {divergence_threshold}m')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # RPE per segment
    axes[1].bar(range(len(rpe_errors)), rpe_errors, color='steelblue', alpha=0.7)
    axes[1].set_xlabel(f'Segment (each = {rpe_segment_len} frames)')
    axes[1].set_ylabel('Translation error (m)')
    axes[1].set_title(f'RPE per segment  |  mean: {rpe_errors.mean():.4f}m  |  max: {rpe_errors.max():.4f}m')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_plot = os.path.join(cfg.output_path, f"trajectory_eval_{ts}.png")
    plt.savefig(out_plot, dpi=150)
    print(f"Plot saved to: {out_plot}")
    plt.show()


if __name__ == "__main__":
    main()
