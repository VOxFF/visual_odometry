import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def _setup_external_paths(base_dir):
    import sys
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
import cv2
import numpy as np
import pandas as pd

from config.config import Config
from modules.stereo.stereo_interfaces import StereoParamsInterface
from modules.stereo.stereo_depth import StereoDepth
from modules.stereo.stereo_params_YAML import StereoParamsYAML
from modules.stereo.stereo_rectification import StereoRectification
from modules.stereo.stereo_disparity_RAFT import DisparityRAFT
from modules.landmarks.landmark_map import LandmarkMap, compute_covariance, backproject
from pipeline.visualization.video_composition import make_stacked_video

# ── Knobs ─────────────────────────────────────────────────────────────────────
render_images = True
compose_movie = True
# ──────────────────────────────────────────────────────────────────────────────

COLOR_DETECTION  = (0,   255,  0)    # green  — SIFT keypoints with valid depth
COLOR_MAP_PROJ   = (255, 100,  0)    # blue   — map landmarks projected into frame
COLOR_INLIER     = (0,   255,  0)    # green  — PnP inlier correspondence line
COLOR_OUTLIER    = (0,    0,  255)   # red    — PnP outlier correspondence line
COLOR_NO_PNP     = (128, 128,  0)    # yellow — match but PnP failed entirely


def project_landmarks(landmarks, R_world_to_cam, t_world_to_cam, fx, fy, cx, cy, W, H, margin=50):
    """Project visible map landmarks into the image. Returns list of (lm_id, u, v)."""
    result = []
    for lm_id, lm in landmarks.items():
        x_cam = R_world_to_cam @ lm.xyz_world + t_world_to_cam
        if x_cam[2] <= 0:
            continue
        u = fx * x_cam[0] / x_cam[2] + cx
        v = fy * x_cam[1] / x_cam[2] + cy
        if -margin <= u < W + margin and -margin <= v < H + margin:
            result.append((lm_id, u, v))
    return result


def solve_pnp(pts_3d, pts_2d, K):
    if len(pts_3d) < 6:
        return None, None, None
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d.astype(np.float64), pts_2d.astype(np.float64), K, np.zeros(4),
        reprojectionError=4.0, confidence=0.99, iterationsCount=200,
    )
    if not success or inliers is None or len(inliers) < 6:
        return None, None, None
    R, _ = cv2.Rodrigues(rvec)
    mask = np.zeros(len(pts_3d), dtype=bool)
    mask[inliers.flatten()] = True
    return R, tvec.flatten(), mask


def render_frame(img_rect, kp_uv, valid_depth,
                 map_proj,        # list of (lm_id, u, v) — projected map points
                 pts_2d,          # (M,2) matched 2D keypoints
                 lm_ids,          # (M,) matched landmark ids
                 kp_indices,      # (M,) which kp each match used
                 inlier_mask,     # (M,) bool or None
                 lm_id_to_proj,   # dict lm_id → (u,v) in image
                 frame_idx, map_size):

    vis = cv2.cvtColor(img_rect, cv2.COLOR_GRAY2BGR)

    # ── Layer 1: projected map landmarks ──────────────────────────────────────
    for lm_id, u, v in map_proj:
        if 0 <= int(u) < vis.shape[1] and 0 <= int(v) < vis.shape[0]:
            cv2.drawMarker(vis, (int(u), int(v)), COLOR_MAP_PROJ,
                           cv2.MARKER_CROSS, markerSize=8, thickness=1)

    # ── Layer 2: correspondence lines ─────────────────────────────────────────
    for j in range(len(pts_2d)):
        kp_pt = (int(kp_uv[kp_indices[j], 0]), int(kp_uv[kp_indices[j], 1]))
        lm_uv = lm_id_to_proj.get(lm_ids[j])
        if lm_uv is None:
            continue
        lm_pt = (int(lm_uv[0]), int(lm_uv[1]))
        if inlier_mask is not None:
            color = COLOR_INLIER if inlier_mask[j] else COLOR_OUTLIER
        else:
            color = COLOR_NO_PNP
        cv2.line(vis, kp_pt, lm_pt, color, thickness=1)

    # ── Layer 3: SIFT detections with valid depth ──────────────────────────────
    for idx, (u, v) in enumerate(kp_uv):
        if valid_depth[idx]:
            cv2.circle(vis, (int(u), int(v)), 3, COLOR_DETECTION, -1)

    # ── HUD ───────────────────────────────────────────────────────────────────
    n_match   = len(pts_2d)
    n_inlier  = int(inlier_mask.sum()) if inlier_mask is not None else 0
    n_visible = len(map_proj)
    cv2.putText(vis, f"frame {frame_idx}  map:{map_size}  proj:{n_visible}  "
                     f"match:{n_match}  inlier:{n_inlier}",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    stereo_ckpt = cfg.stereo_checkpoint if os.path.isabs(cfg.stereo_checkpoint) \
                  else os.path.join(project_root, cfg.stereo_checkpoint)

    params        = StereoParamsYAML(cfg.yaml_file)
    rectification = StereoRectification(params)
    disp_solver   = DisparityRAFT(stereo_ckpt, rectification, cfg.raft_iters, cfg.raft_disparity_warmstart)
    depth_solver  = StereoDepth(params)

    stereo_mask, _, _, _ = rectification.get_rectification_masks()
    cam_params = params.get_camera_params(StereoParamsInterface.StereoCamera.LEFT)
    K_raw = cam_params.get_intrinsics()
    fx, fy = float(K_raw[0, 0]), float(K_raw[1, 1])
    cx, cy = float(K_raw[0, 2]), float(K_raw[1, 2])
    baseline = float(params.get_baseline())
    K = K_raw.astype(np.float64)

    sift_kernel = np.ones((11, 11), np.uint8)
    sift_mask   = cv2.erode(stereo_mask.astype(np.uint8) * 255, sift_kernel, iterations=1)
    sift        = cv2.SIFT_create(nfeatures=cfg.max_keypoints)
    lm_map      = LandmarkMap(match_ratio=0.75, max_merge_dist_3d=0.5)

    left_txt  = os.path.join(cfg.dataset_path, "left_images.txt")
    df_left   = pd.read_csv(left_txt, sep=r'\s+', comment='#', names=['id', 'timestamp', 'image_name'])
    all_files = df_left['image_name'].tolist()
    left_files = all_files[cfg.start_frame:]
    if cfg.limit:
        left_files = left_files[:cfg.limit]

    out_dir = os.path.join(cfg.output_path, "out_lm_kp")

    if render_images:
        os.makedirs(out_dir, exist_ok=True)

        R_prev, t_prev = np.eye(3), np.zeros(3)
        first_frame = True
        H, W = stereo_mask.shape

        for i, left_rel in enumerate(left_files):
            frame_idx = i + cfg.start_frame

            left_path  = os.path.join(cfg.dataset_path, left_rel)
            right_path = left_path.replace("image_0_", "image_1_")
            img_left  = cv2.imread(left_path,  cv2.IMREAD_GRAYSCALE)
            img_right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
            if img_left is None or img_right is None:
                continue

            img_left_rect, _ = rectification.rectify_images(img_left, img_right)
            disp  = disp_solver.compute_disparity(img_left, img_right)
            depth = depth_solver.compute_depth(disp)

            kps, descs = sift.detectAndCompute(img_left_rect, sift_mask)
            if descs is None or len(kps) == 0:
                cv2.imwrite(os.path.join(out_dir, f"{frame_idx:06d}.png"),
                            cv2.cvtColor(img_left_rect, cv2.COLOR_GRAY2BGR))
                continue

            kp_uv = np.array([[k.pt[0], k.pt[1]] for k in kps], dtype=np.float32)
            us = np.clip(kp_uv[:, 0].astype(int), 0, W - 1)
            vs = np.clip(kp_uv[:, 1].astype(int), 0, H - 1)
            z_vals    = depth[vs, us]
            valid_dep = (z_vals >= cfg.min_depth) & (z_vals <= cfg.max_depth) & (z_vals > 0)

            if first_frame:
                # Initialise map — no correspondences to show yet
                for idx in range(len(kp_uv)):
                    if not valid_dep[idx]:
                        continue
                    xyz_cam = backproject(np.array([kp_uv[idx, 0]]), np.array([kp_uv[idx, 1]]),
                                          np.array([z_vals[idx]]), fx, fy, cx, cy)[0]
                    cov = compute_covariance(z_vals[idx], fx, fy, baseline, np.eye(3))
                    lm_map.add(xyz_cam, cov, descs[idx], frame_idx)
                vis = render_frame(img_left_rect, kp_uv, valid_dep,
                                   [], np.empty((0,2)), np.empty(0,int),
                                   np.empty(0,int), None, {}, frame_idx, lm_map.size)
                cv2.imwrite(os.path.join(out_dir, f"{frame_idx:06d}.png"), vis)
                first_frame = False
                if frame_idx % 20 == 0:
                    print(f"[{frame_idx}] map:{lm_map.size}")
                continue

            # ── Project visible map landmarks for display ──────────────────
            map_proj     = project_landmarks(lm_map.landmarks, R_prev, t_prev,
                                             fx, fy, cx, cy, W, H)
            lm_id_to_proj = {lid: (u, v) for lid, u, v in map_proj}
            visible_ids   = [lid for lid, _, _ in map_proj]

            # ── Match + PnP ────────────────────────────────────────────────
            pts_3d, pts_2d, lm_ids, kp_indices = lm_map.get_correspondences(
                kp_uv, descs, visible_ids=visible_ids)

            R_curr, t_curr, inlier_mask = solve_pnp(pts_3d, pts_2d, K)

            vis = render_frame(img_left_rect, kp_uv, valid_dep,
                               map_proj, pts_2d, lm_ids, kp_indices,
                               inlier_mask, lm_id_to_proj, frame_idx, lm_map.size)
            cv2.imwrite(os.path.join(out_dir, f"{frame_idx:06d}.png"), vis)

            # ── Update map ─────────────────────────────────────────────────
            pose_R = R_curr if R_curr is not None else R_prev
            pose_t = t_curr if t_curr is not None else t_prev

            matched_kp = set()
            if R_curr is not None and inlier_mask is not None:
                for j, lm_id in enumerate(lm_ids):
                    if inlier_mask[j]:
                        matched_kp.add(int(kp_indices[j]))
                        x_cam = pose_R @ lm_map.landmarks[lm_id].xyz_world + pose_t
                        if x_cam[2] > 0:
                            R_c2w = pose_R.T
                            xyz_w = R_c2w @ (backproject(
                                np.array([kp_uv[kp_indices[j],0]]),
                                np.array([kp_uv[kp_indices[j],1]]),
                                np.array([z_vals[kp_indices[j]]]), fx, fy, cx, cy)[0]) \
                                + (-R_c2w @ pose_t)
                            cov = compute_covariance(z_vals[kp_indices[j]], fx, fy, baseline, R_c2w)
                            lm_map.merge(lm_id, xyz_w, cov, descs[kp_indices[j]], frame_idx)

            for idx in range(len(kp_uv)):
                if idx in matched_kp or not valid_dep[idx]:
                    continue
                R_c2w = pose_R.T
                xyz_cam = backproject(np.array([kp_uv[idx,0]]), np.array([kp_uv[idx,1]]),
                                      np.array([z_vals[idx]]), fx, fy, cx, cy)[0]
                xyz_w = R_c2w @ xyz_cam + (-R_c2w @ pose_t)
                cov   = compute_covariance(z_vals[idx], fx, fy, baseline, R_c2w)
                lm_map.add(xyz_w, cov, descs[idx], frame_idx)

            if R_curr is not None:
                R_prev, t_prev = R_curr.copy(), t_curr.copy()

            if frame_idx % 20 == 0:
                n_in = int(inlier_mask.sum()) if inlier_mask is not None else 0
                print(f"[{frame_idx}] map:{lm_map.size}  corr:{len(pts_2d)}  inlier:{n_in}")

    if compose_movie:
        print("Composing movie...")
        rel_out = os.path.relpath(cfg.output_path, cfg.dataset_path)
        frame_start = cfg.start_frame
        transformations = [
            lambda x: os.path.join(rel_out, "out_lm_kp",
                                   f"{int(x.split('_')[-1].split('.')[0]) :06d}.png"),
        ]
        make_stacked_video(cfg.dataset_path, left_files,
                           os.path.join(cfg.output_path, "landmarks_kp_video.mp4"),
                           transformations)

    print("Done.")


if __name__ == "__main__":
    main()
