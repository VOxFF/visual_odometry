"""
FlowLandmarkPipeline
====================
Frame-to-frame tracking via RAFT dense optical flow + persistent 3D landmark
map + PnP pose estimation.

Tracking:
    RAFT computes dense (dx, dy) per pixel. Tracked keypoint positions are
    propagated by sampling the flow field at their pixel coordinates.
    A forward-backward consistency check filters unreliable tracks.

Association:
    Done in 3D world space. New keypoints are back-projected to 3D and
    suppressed if an existing active landmark lies within BIRTH_RADIUS_3D
    metres — no 2D proximity heuristics needed.

Map lifecycle:
    tentative (obs_count < CONFIRM_OBS) → active → culled
    Tentative landmarks are tracked but excluded from PnP.
    Culling removes unconfirmed and stale active landmarks.
"""

import os
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

from config.config import Config
from pipeline.base_pipeline import PipelineBase
from modules.landmarks.landmark_map_guided import GuidedLandmarkMap
from modules.landmarks.landmark_map import compute_covariance, backproject
from modules.landmarks.h5_export import LandmarkH5Exporter
from modules.flow.flow_map_RAFT import OpticalFlowRAFT


# ── Parameters ────────────────────────────────────────────────────────────────
BIRTH_RADIUS_3D   = 0.5   # m:  suppress new landmark if active one within this
CONFIRM_OBS       = 3     # PnP inlier observations before tentative → active
MAX_TENTATIVE_AGE = 5     # frames: drop tentative if not re-tracked in this window
CULL_STALE_AFTER  = 50    # frames: drop active landmark not seen as inlier for this long
MAX_PNP_MISSES    = 3     # candidate PnP outliers before a tracked landmark is culled
MIN_PNP_INLIERS  = 50
MIN_PNP_INLIER_RATIO = 0.10
DEPTH_PATCH_HALF  = 3     # px:  half-size of depth quality patch (7×7 total)
DEPTH_MAX_SPREAD  = 0.5   # m:  max IQR of valid depths in patch
DEPTH_MIN_VALID   = 0.5   # fraction: min valid pixels in patch
FB_THRESH         = 5.0   # px: forward-backward flow consistency threshold
# ──────────────────────────────────────────────────────────────────────────────


class FlowLandmarkPipeline(PipelineBase):

    def __init__(self, config: Config):
        super().__init__(config)

        # RAFT flow solver
        base = os.path.dirname(os.path.dirname(__file__))
        flow_ckpt = self.cfg.flow_checkpoint if os.path.isabs(self.cfg.flow_checkpoint) \
                    else os.path.join(base, self.cfg.flow_checkpoint)
        # Pass rectification=None — we rectify manually so flow runs on rectified images
        # Warm-start is disabled here because this pipeline computes both
        # forward and backward flow every frame; alternating directions would
        # make a single warm-start state invalid.
        self.flow_solver = OpticalFlowRAFT(flow_ckpt, None, self.cfg.raft_iters, False)

        self.map      = GuidedLandmarkMap(max_merge_dist_3d=0.5)
        self.exporter = LandmarkH5Exporter()

        # Detection mask: 5px eroded stereo valid region
        k = np.ones((11, 11), np.uint8)
        self.det_mask = cv2.erode(
            self.stereo_mask.astype(np.uint8) * 255, k, iterations=1)

        K = self.cam_params.get_intrinsics()
        self.fx = float(K[0, 0]);  self.fy = float(K[1, 1])
        self.cx = float(K[0, 2]);  self.cy = float(K[1, 2])
        self.baseline = float(self.params.get_baseline())
        self.K = K.astype(np.float64)

        # Tracking state
        # tracked_pts : (N, 2) float32 — current 2D positions of all tracked points
        # tracked_ids : [int]*N        — parallel landmark ids
        # obs_count   : {lm_id: int}  — successful PnP inlier observations
        # born_frame  : {lm_id: int}  — frame when landmark was added
        # last_seen   : {lm_id: int}  — last frame where landmark was a PnP inlier
        # pnp_misses  : {lm_id: int}  — consecutive PnP candidate outlier count
        self.tracked_pts: np.ndarray     = np.empty((0, 2), dtype=np.float32)
        self.tracked_ids: list[int]      = []
        self.obs_count:   dict[int, int] = {}
        self.born_frame:  dict[int, int] = {}
        self.last_seen:   dict[int, int] = {}
        self.pnp_misses:  dict[int, int] = {}

        self._prev_rect: np.ndarray | None = None
        self._pos_tree:  cKDTree | None    = None   # lazy 3D index of active landmarks

    # ── Main loop ──────────────────────────────────────────────────────────────

    def compute_trajectory(self):
        print("Computing trajectory (flow-landmark pipeline)...")

        R_prev, t_prev = np.eye(3), np.zeros(3)
        first_frame = True
        t_load = t_disp = t_flow = t_track = t_detect = t_cull = 0.0
        _tick = time.perf_counter

        with open(self.traj_txt_path, 'w') as f:
            f.write("frame, translation, rotation_matrix_flat\n")

            for i, left_rel in enumerate(self.left_files):
                local_frame_idx = i
                frame_idx = local_frame_idx + self.frame_start

                t0 = _tick()
                img_left, img_right = self._load_stereo(left_rel)
                if img_left is None or img_right is None:
                    continue
                t_load += _tick() - t0

                t0 = _tick()
                img_rect, _ = self.rectification.rectify_images(img_left, img_right)
                disp  = self.disparity_solver.compute_disparity(img_left, img_right)
                depth = self.depth_solver.compute_depth(disp)
                t_disp += _tick() - t0

                H, W = img_rect.shape

                print(f"[{frame_idx}] img={img_rect.shape} depth_valid={np.sum((depth>0)&(depth<self.cfg.max_depth))}", flush=True)

                # ── First frame: seed the map ──────────────────────────────
                if first_frame:
                    self._prev_rect = img_rect
                    new_xyzs = self._detect_and_add(
                        img_rect, depth, R_prev, t_prev, frame_idx)
                    self.exporter.record_new(local_frame_idx, new_xyzs)
                    first_frame = False
                    print(f"[{frame_idx}] first frame: seeded {len(new_xyzs)} landmarks", flush=True)
                    f.write(
                        f"{frame_idx}, {[0.,0.,0.]}, {np.eye(3).flatten().tolist()}\n")
                    continue

                # ── RAFT flow ──────────────────────────────────────────────
                t0 = _tick()
                flow_fwd = self.flow_solver.compute_flow(self._prev_rect, img_rect)
                flow_bwd = self.flow_solver.compute_flow(img_rect, self._prev_rect)
                t_flow += _tick() - t0

                # ── Track + PnP ────────────────────────────────────────────
                t0 = _tick()
                R_curr, t_curr, inlier_ids, pose_healthy = self._track_and_pnp(
                    flow_fwd, flow_bwd, img_rect, depth, frame_idx)
                t_track += _tick() - t0

                if R_curr is None or not pose_healthy:
                    f.write(
                        f"{frame_idx}, {[0.,0.,0.]}, {np.eye(3).flatten().tolist()}\n")
                    pose_R, pose_t = R_prev, t_prev
                    if R_curr is not None:
                        print("  DBG pose: rejected unhealthy PnP result", flush=True)
                else:
                    R_rel = R_curr @ R_prev.T
                    t_rel = t_curr - R_rel @ t_prev
                    print(
                        f"  DBG pose: t_abs={np.linalg.norm(t_curr):.4f}m "
                        f"t_rel={np.linalg.norm(t_rel):.4f}m",
                        flush=True,
                    )
                    f.write(
                        f"{frame_idx}, {t_rel.tolist()}, {R_rel.flatten().tolist()}\n")
                    R_prev, t_prev = R_curr.copy(), t_curr.copy()
                    pose_R, pose_t = R_curr, t_curr

                # ── Detect new landmarks ───────────────────────────────────
                t0 = _tick()
                if not pose_healthy:
                    self._reset_tracks()
                    print("  DBG detect: reset/reseed after unhealthy pose", flush=True)
                new_xyzs = self._detect_and_add(
                    img_rect, depth, pose_R, pose_t, frame_idx)
                self.exporter.record_new(local_frame_idx, new_xyzs)
                t_detect += _tick() - t0

                # ── Cull stale landmarks ───────────────────────────────────
                t0 = _tick()
                self._cull(frame_idx)
                t_cull += _tick() - t0

                self._prev_rect = img_rect

                if frame_idx % 20 == 0 and i > 0:
                    n_active = sum(1 for lid in self.tracked_ids
                                   if self.obs_count.get(lid, 0) >= CONFIRM_OBS)
                    n_tent   = len(self.tracked_ids) - n_active
                    print(
                        f"[{frame_idx:4d}/{len(self.left_files)}]"
                        f"  map:{self.map.size:5d}"
                        f"  active:{n_active:4d}  tent:{n_tent:3d}"
                        f"  load:{t_load*1e3/20:5.1f}ms"
                        f"  disp:{t_disp*1e3/20:5.1f}ms"
                        f"  flow:{t_flow*1e3/20:5.1f}ms"
                        f"  track:{t_track*1e3/20:5.1f}ms"
                        f"  detect:{t_detect*1e3/20:5.1f}ms"
                        f"  cull:{t_cull*1e3/20:5.1f}ms",
                        flush=True,
                    )
                    t_load = t_disp = t_flow = t_track = t_detect = t_cull = 0.0

        print("Trajectory complete:", self.traj_txt_path)
        ts = os.path.splitext(os.path.basename(self.traj_txt_path))[0].split("_", 2)[2]
        self.exporter.write(
            os.path.join(self.cfg.output_path, f"landmarks_{ts}.h5"))

    # ── RAFT flow track → PnP → merge ─────────────────────────────────────────

    def _track_and_pnp(self, flow_fwd, flow_bwd, img_rect, depth, frame_idx):
        """
        Propagate tracked points via RAFT flow, filter by forward-backward
        consistency, run PnP on confirmed tracks, merge inliers into 3D map.

        Returns (R_curr, t_curr, inlier_lm_ids, pose_healthy).
        """
        if len(self.tracked_ids) == 0:
            return None, None, [], False

        H, W = img_rect.shape

        # ── Sample RAFT flow at tracked positions ──────────────────────────
        us = self.tracked_pts[:, 0].astype(int).clip(0, W - 1)
        vs = self.tracked_pts[:, 1].astype(int).clip(0, H - 1)
        dx = flow_fwd[0, vs, us].astype(np.float32)
        dy = flow_fwd[1, vs, us].astype(np.float32)
        flow_mag = np.linalg.norm(np.stack([dx, dy], axis=1), axis=1)
        print(
            f"  DBG flow_mag: med={np.median(flow_mag):.2f}px "
            f"p95={np.percentile(flow_mag, 95):.2f}px "
            f"max={np.max(flow_mag):.2f}px",
            flush=True,
        )
        next_pts = self.tracked_pts + np.stack([dx, dy], axis=1)

        # ── Forward-backward consistency: sample true reverse flow at next_pts ──
        us2 = next_pts[:, 0].astype(int).clip(0, W - 1)
        vs2 = next_pts[:, 1].astype(int).clip(0, H - 1)
        dx2 = flow_bwd[0, vs2, us2].astype(np.float32)
        dy2 = flow_bwd[1, vs2, us2].astype(np.float32)
        back_pts = next_pts + np.stack([dx2, dy2], axis=1)

        fb_err = np.linalg.norm(self.tracked_pts - back_pts, axis=1)

        in_frame = ((next_pts[:, 0] >= 0) & (next_pts[:, 0] < W) &
                    (next_pts[:, 1] >= 0) & (next_pts[:, 1] < H))
        valid = (fb_err < FB_THRESH) & in_frame

        # Update tracked state
        kept_ids = [self.tracked_ids[j] for j in range(len(self.tracked_ids)) if valid[j]]
        kept_pts = next_pts[valid].astype(np.float32)

        print(f"  DBG flow: total={len(self.tracked_ids)}"
              f" fb_ok={valid.sum()} (fb_mean={fb_err.mean():.2f}px)"
              f" in_frame={in_frame.sum()} kept={len(kept_ids)}", flush=True)

        self.tracked_ids = kept_ids
        self.tracked_pts = kept_pts

        if len(kept_ids) == 0:
            return None, None, [], False

        # ── PnP on live tracks with valid current-frame depth. PnP inliers
        # promote tentative landmarks.
        current_depths = np.array([
            self._sample_depth(depth, pt[0], pt[1]) or np.nan
            for pt in kept_pts
        ])
        depth_mask = np.isfinite(current_depths)
        conf_mask = np.array([
            lid in self.map.landmarks
            for lid in kept_ids], dtype=bool) & depth_mask

        active_candidates = sum(
            1 for j, lid in enumerate(kept_ids)
            if conf_mask[j] and self.obs_count.get(lid, 0) >= CONFIRM_OBS
        )
        print(
            f"  DBG pnp:  candidates={conf_mask.sum()} active={active_candidates} "
            f"depth_ok={depth_mask.sum()}",
            flush=True,
        )

        if conf_mask.sum() < 6:
            return None, None, [], False

        conf_ids = [kept_ids[j] for j in range(len(kept_ids)) if conf_mask[j]]
        conf_pts = kept_pts[conf_mask]
        pts_3d   = np.array([self.map.landmarks[lid].xyz_world for lid in conf_ids])
        z3d = current_depths[conf_mask]
        print(
            f"  DBG pnp_depth: med={np.median(z3d):.2f}m "
            f"min={np.min(z3d):.2f}m max={np.max(z3d):.2f}m",
            flush=True,
        )

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, conf_pts.astype(np.float64), self.K, np.zeros(4),
            reprojectionError=4.0, confidence=0.99, iterationsCount=200)

        print(f"  DBG pnp:  success={success}"
              f" inliers={len(inliers) if inliers is not None else 0}", flush=True)

        if not success or inliers is None or len(inliers) < 6:
            for lid in conf_ids:
                self.pnp_misses[lid] = self.pnp_misses.get(lid, 0) + 1
            return None, None, [], False

        R_curr   = cv2.Rodrigues(rvec)[0]
        t_curr   = tvec.flatten()
        inlier_flat = inliers.flatten()
        inlier_ids  = [conf_ids[j] for j in inlier_flat]
        inlier_pts  = conf_pts[inlier_flat]
        inlier_set = set(inlier_ids)
        inlier_ratio = len(inlier_ids) / len(conf_ids)
        pose_healthy = (
            len(inlier_ids) >= MIN_PNP_INLIERS
            and inlier_ratio >= MIN_PNP_INLIER_RATIO
        )
        print(
            f"  DBG pnp:  inlier_ratio={inlier_ratio:.2f} "
            f"healthy={pose_healthy}",
            flush=True,
        )

        for lid in conf_ids:
            if lid in inlier_set:
                self.obs_count[lid] = self.obs_count.get(lid, 0) + 1
                self.pnp_misses[lid] = 0
            else:
                self.pnp_misses[lid] = self.pnp_misses.get(lid, 0) + 1

        # ── Merge inliers into 3D map (information filter) ────────────────
        R_c2w = R_curr.T
        t_c2w = -R_c2w @ t_curr

        for lid, uv in zip(inlier_ids, inlier_pts):
            z = self._sample_depth(depth, uv[0], uv[1])
            if z is None:
                continue
            xyz_cam = backproject(
                np.array([uv[0]]), np.array([uv[1]]), np.array([z]),
                self.fx, self.fy, self.cx, self.cy)[0]
            xyz_w = R_c2w @ xyz_cam + t_c2w
            if not self._world_point_valid_for_pose(xyz_w, R_curr, t_curr):
                continue
            cov   = compute_covariance(z, self.fx, self.fy, self.baseline, R_c2w)
            self.map.merge(
                lid, xyz_w, cov,
                self.map._lm_to_desc.get(lid, np.zeros(128, np.float32)),
                frame_idx)
            self.last_seen[lid] = frame_idx

        self._pos_tree = None   # invalidate 3D index after merges
        return R_curr, t_curr, inlier_ids, pose_healthy

    # ── Detect + suppress + add ────────────────────────────────────────────────

    def _detect_and_add(self, img_rect, depth, R_w2c, t_w2c, frame_idx):
        """
        Detect Shi-Tomasi keypoints, apply 3D birth suppression and depth
        patch filter, start tracking survivors as tentative landmarks.
        Returns list of new world xyz positions (for H5 export).
        """
        corners = cv2.goodFeaturesToTrack(
            img_rect,
            maxCorners=self.cfg.max_keypoints,
            qualityLevel=0.01,
            minDistance=8,
            mask=self.det_mask,
        )
        print(f"  DBG detect: corners={'None' if corners is None else len(corners)}", flush=True)
        if corners is None:
            return []

        candidates = corners.reshape(-1, 2)

        # ── 2D suppression: near currently tracked points ──────────────────
        if len(self.tracked_pts) > 0:
            tree_2d = cKDTree(self.tracked_pts)
            dists, _ = tree_2d.query(candidates, k=1)
            candidates = candidates[dists > 8.0]

        print(f"  DBG detect: after_2d_suppress={len(candidates)}", flush=True)
        if len(candidates) == 0:
            return []

        # ── Depth patch filter + backproject candidates ────────────────────
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c
        backprojected = []

        for uv in candidates:
            z = self._sample_depth(depth, uv[0], uv[1])
            if z is None:
                continue
            xyz_cam = backproject(
                np.array([uv[0]]), np.array([uv[1]]), np.array([z]),
                self.fx, self.fy, self.cx, self.cy)[0]
            xyz_w = R_c2w @ xyz_cam + t_c2w
            if not self._world_point_valid_for_pose(xyz_w, R_w2c, t_w2c):
                continue
            backprojected.append((uv, z, xyz_w))

        print(f"  DBG detect: backproj={len(backprojected)}/{len(candidates)}", flush=True)
        if not backprojected:
            return []

        # ── 3D birth suppression ───────────────────────────────────────────
        self._rebuild_pos_tree()
        survived = []
        for uv, z, xyz_w in backprojected:
            if self._pos_tree is not None:
                d3, _ = self._pos_tree.query(xyz_w, k=1)
                if d3 < BIRTH_RADIUS_3D:
                    continue
            survived.append((uv, z, xyz_w))

        if not survived:
            return []

        # ── Add survivors as tentative landmarks ───────────────────────────
        new_world_xyzs = []
        for uv, z, xyz_w in survived:
            cov   = compute_covariance(z, self.fx, self.fy, self.baseline, R_c2w)
            lm_id = self.map.add(xyz_w, cov, np.zeros(128, np.float32), frame_idx)

            self.obs_count[lm_id]  = 1
            self.born_frame[lm_id] = frame_idx
            self.last_seen[lm_id]  = frame_idx
            self.pnp_misses[lm_id] = 0

            pt = np.array([[uv[0], uv[1]]], dtype=np.float32)
            self.tracked_pts = np.vstack([self.tracked_pts, pt]) \
                               if len(self.tracked_pts) > 0 else pt
            self.tracked_ids.append(lm_id)
            new_world_xyzs.append(xyz_w)

        self._pos_tree = None   # invalidate after additions
        print(f"  DBG detect: backproj={len(backprojected)}"
              f" survived_3d={len(survived)}"
              f" added={len(new_world_xyzs)}", flush=True)
        return new_world_xyzs

    # ── Culling ────────────────────────────────────────────────────────────────

    def _cull(self, frame_idx):
        to_remove = set()
        for lid in list(self.obs_count):
            age  = frame_idx - self.born_frame.get(lid, frame_idx)
            seen = self.last_seen.get(lid, frame_idx)
            if self.obs_count[lid] < CONFIRM_OBS and age > MAX_TENTATIVE_AGE:
                to_remove.add(lid)
            elif self.obs_count[lid] >= CONFIRM_OBS and \
                    (frame_idx - seen) > CULL_STALE_AFTER:
                to_remove.add(lid)
            elif self.pnp_misses.get(lid, 0) > MAX_PNP_MISSES:
                to_remove.add(lid)

        if not to_remove:
            return

        keep = [j for j, lid in enumerate(self.tracked_ids) if lid not in to_remove]
        self.tracked_ids = [self.tracked_ids[j] for j in keep]
        self.tracked_pts = self.tracked_pts[keep] \
                           if keep else np.empty((0, 2), dtype=np.float32)

        for lid in to_remove:
            self.map.landmarks.pop(lid, None)
            self.map._lm_to_desc.pop(lid, None)
            self.obs_count.pop(lid, None)
            self.born_frame.pop(lid, None)
            self.last_seen.pop(lid, None)
            self.pnp_misses.pop(lid, None)

        self.map._dirty = True
        self._pos_tree  = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _reset_tracks(self):
        self.tracked_pts = np.empty((0, 2), dtype=np.float32)
        self.tracked_ids = []
        self.map.landmarks.clear()
        self.map._lm_to_desc.clear()
        self.map._lm_index_ids.clear()
        self.map._desc_matrix = None
        self.map._dirty = True
        self.obs_count.clear()
        self.born_frame.clear()
        self.last_seen.clear()
        self.pnp_misses.clear()
        self._pos_tree = None

    def _sample_depth(self, depth: np.ndarray, u: float, v: float):
        """
        Median depth in a patch around (u, v) with quality checks.
        Returns float metres or None if patch is too noisy / sparse.
        """
        H, W = depth.shape
        r  = DEPTH_PATCH_HALF
        u0, u1 = max(0, int(u) - r), min(W, int(u) + r + 1)
        v0, v1 = max(0, int(v) - r), min(H, int(v) + r + 1)
        patch = depth[v0:v1, u0:u1]
        valid = patch[(patch >= self.cfg.min_depth) & (patch <= self.cfg.max_depth)]
        if valid.size == 0 or valid.size / patch.size < DEPTH_MIN_VALID:
            return None
        if float(np.percentile(valid, 75) - np.percentile(valid, 25)) > DEPTH_MAX_SPREAD:
            return None
        return float(np.median(valid))

    def _rebuild_pos_tree(self):
        """Lazily rebuild 3D kd-tree of reliable active landmark positions."""
        if self._pos_tree is not None:
            return
        if not self.map.landmarks:
            return
        ids = [
            lid for lid in self.map.landmarks
            if self.obs_count.get(lid, 0) >= CONFIRM_OBS
            and self.pnp_misses.get(lid, 0) == 0
        ]
        if not ids:
            return
        xyzs = np.array([self.map.landmarks[i].xyz_world for i in ids])
        self._pos_tree = cKDTree(xyzs)

    def _world_point_valid_for_pose(self, xyz_w: np.ndarray,
                                    R_w2c: np.ndarray,
                                    t_w2c: np.ndarray) -> bool:
        xyz_cam = R_w2c @ xyz_w + t_w2c
        z = float(xyz_cam[2])
        return self.cfg.min_depth <= z <= self.cfg.max_depth
