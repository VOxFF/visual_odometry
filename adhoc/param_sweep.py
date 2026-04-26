
import os
import sys
import copy
import itertools
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def _setup_external_paths(base_dir: str):
    raft_stereo_path = os.path.join(base_dir, "external", "RAFT-Stereo")
    raft_flow_path   = os.path.join(base_dir, "external", "RAFT-Flow")
    core_path        = os.path.join(raft_flow_path, "flow_core")
    aanet_path       = os.path.join(base_dir, "external", "aanet")
    for p in [raft_stereo_path, raft_flow_path, aanet_path]:
        if p not in sys.path:
            sys.path.append(p)
    if core_path not in sys.path:
        sys.path.insert(0, core_path)

_setup_external_paths(project_root)

import argparse
import numpy as np

from config.config import Config
from pipeline.pipeline import CameraTrackingPipeline
from adhoc.eval_trajectory import (
    load_estimated_trajectory,
    compute_ate,
    compute_rpe,
    find_divergence_frame,
)
from modules.imu.imu_YAML import YamlIMU
from modules.io.data_utils import match_ground_truth_positions

# ── Sweep grid ─────────────────────────────────────────────────────────────────
# Each key must match a Config field name.
# All combinations will be run.
SWEEP_GRID = {
    'dz_threshold':     [0.5, 1.0, 2.0],
    'log_dz_threshold': [False, True],
    'max_keypoints':    [320],
}

DIVERGENCE_THRESHOLD = 1.0   # metres (for eval)
RPE_SEGMENT_LEN      = 10    # frames
# ──────────────────────────────────────────────────────────────────────────────


def run_eval(cfg: Config, traj_path: str) -> dict:
    gt_path  = os.path.join(cfg.dataset_path, "groundtruth.txt")
    left_txt = os.path.join(cfg.dataset_path, "left_images.txt")

    est_positions = load_estimated_trajectory(traj_path, cfg.limit)
    imu = YamlIMU(cfg.yaml_file, left_txt, gt_path)
    R_cam_in_world = imu.get_initial_pose()
    est_world = [R_cam_in_world @ p for p in est_positions]

    combined = match_ground_truth_positions(est_world, left_txt, gt_path)
    valid = [(est, gt) for est, gt in combined if gt is not None]
    if not valid:
        return {}

    est_matched = [p for p, _ in valid]
    gt_matched  = [p for _, p in valid]

    pre_drift = float(np.linalg.norm(np.array(est_matched[0]) - np.array(gt_matched[0])))
    per_frame_errors, ate_rmse = compute_ate(est_matched, gt_matched)
    rpe_errors  = compute_rpe(est_matched, gt_matched, RPE_SEGMENT_LEN)
    div_frame   = find_divergence_frame(per_frame_errors, DIVERGENCE_THRESHOLD)

    return {
        'frames':      len(per_frame_errors),
        'pre_drift':   pre_drift,
        'ate_rmse':    ate_rmse,
        'ate_mean':    float(per_frame_errors.mean()),
        'ate_max':     float(per_frame_errors.max()),
        'rpe_mean':    float(rpe_errors.mean()),
        'rpe_max':     float(rpe_errors.max()),
        'pct_good':    float(np.mean(per_frame_errors < DIVERGENCE_THRESHOLD) * 100),
        'div_frame':   div_frame,
    }


def label_for(params: dict) -> str:
    parts = []
    for k, v in params.items():
        short = {
            'dz_threshold':     'dz',
            'log_dz_threshold': 'logdz',
            'max_keypoints':    'kp',
        }.get(k, k)
        if isinstance(v, bool):
            parts.append(f"{short}={'T' if v else 'F'}")
        else:
            parts.append(f"{short}={v}")
    return '_'.join(parts)


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep — trajectory only, no video")
    parser.add_argument("--config", required=True, help="Base YAML config file")
    args = parser.parse_args()

    base_cfg = Config.from_yaml(args.config)
    sweep_root = os.path.join(base_cfg.output_path, "sweep")
    os.makedirs(sweep_root, exist_ok=True)

    # Build all combinations
    keys   = list(SWEEP_GRID.keys())
    values = list(SWEEP_GRID.values())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    total = len(combos)
    print(f"Sweep: {total} combinations")
    print(f"Grid: { {k: v for k, v in SWEEP_GRID.items()} }\n")

    results = []
    sweep_t0 = datetime.now()

    for idx, params in enumerate(combos):
        lbl = label_for(params)
        run_dir = os.path.join(sweep_root, lbl)
        os.makedirs(run_dir, exist_ok=True)
        traj_path = os.path.join(run_dir, "camera_trajectory.txt")

        pct = (idx / total) * 100
        elapsed_so_far = (datetime.now() - sweep_t0).total_seconds()
        eta_str = ""
        if idx > 0:
            avg_per_run = elapsed_so_far / idx
            eta_sec = avg_per_run * (total - idx)
            eta_str = f"  ETA ~{eta_sec/60:.0f} min"

        print(f"{'='*60}", flush=True)
        print(f"Pass {idx+1}/{total}  ({pct:.0f}% sweep done){eta_str}", flush=True)
        print(f"Config: {lbl}", flush=True)
        print(f"{'='*60}", flush=True)

        # Build modified config
        cfg = copy.copy(base_cfg)
        cfg.output_path        = run_dir
        cfg.compute_trajectory = True
        cfg.render_images      = False
        cfg.compose_movie      = False
        for k, v in params.items():
            setattr(cfg, k, v)

        # Run trajectory
        t0 = datetime.now()
        pipeline = CameraTrackingPipeline(cfg)
        pipeline.compute_trajectory()
        elapsed = (datetime.now() - t0).total_seconds()
        print(f"Trajectory done in {elapsed/60:.1f} min", flush=True)

        # Eval
        print("Evaluating...", flush=True)
        metrics = run_eval(cfg, traj_path)
        if not metrics:
            print(f"  -> eval failed (no GT matches)\n", flush=True)
            continue

        metrics['label']   = lbl
        metrics['elapsed'] = elapsed
        metrics.update(params)
        results.append(metrics)

        print(f"  ATE RMSE  : {metrics['ate_rmse']:.4f} m", flush=True)
        print(f"  RPE mean  : {metrics['rpe_mean']:.4f} m", flush=True)
        print(f"  Div frame : {metrics['div_frame']}", flush=True)
        print(flush=True)

    if not results:
        print("No results collected.")
        return

    # ── Summary table ──────────────────────────────────────────────────────────
    results.sort(key=lambda r: r['ate_rmse'])

    header = f"{'Label':<30} {'ATE RMSE':>10} {'ATE mean':>10} {'RPE mean':>10} {'Div frame':>10} {'Time':>8}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in results:
        div = str(r['div_frame']) if r['div_frame'] is not None else "none"
        print(f"{r['label']:<30} {r['ate_rmse']:>10.4f} {r['ate_mean']:>10.4f} "
              f"{r['rpe_mean']:>10.4f} {div:>10} {r['elapsed']/60:>7.1f}m")
    print(sep)

    # Save summary CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(sweep_root, f"sweep_results_{ts}.csv")
    import csv
    fieldnames = ['label'] + keys + ['ate_rmse', 'ate_mean', 'ate_max',
                                      'rpe_mean', 'rpe_max', 'pct_good',
                                      'div_frame', 'pre_drift', 'elapsed']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
