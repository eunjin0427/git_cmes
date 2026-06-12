"""
rule_based_suction.py
─────────────────────────────────────────────────────────
ROI + SAM 마스크 기반 룰베이스 흡착점 선정

흐름:
  1. 전체 depth 이미지에서 ROI 영역만 크롭
  2. ROI depth → Point Cloud 역투영
  3. SAM 마스크로 물체별 포인트 분리
  4. 각 물체별:
     - 노이즈 필터링
     - RANSAC 평면 피팅
     - KNN 법선 추정 + 곡률 계산
     - 물체별 전략으로 파지점 선정

사용:
  from rule_based_suction import SuctionEstimator
  estimator = SuctionEstimator(cam)
  results = estimator.process(depth_img, rgb_img, roi, detections)
"""

import os
import numpy as np
import open3d as o3d
import cv2
import yaml
from typing import Optional, Tuple, List

# ──────────────────────────────────────────────────────────────────────────────
# 경로
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
PICKNPLACE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))


def _load_config() -> dict:
    config_path = os.path.join(PICKNPLACE_DIR, "configs", "picking_config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f.read())


# ──────────────────────────────────────────────────────────────────────────────
# 물체별 파지 전략
# ──────────────────────────────────────────────────────────────────────────────
LABEL_STRATEGY = {
    "haribo"     : {"type": "vinyl",       "planarity_w": 0.6, "normal_w": 0.3, "centroid_w": 0.1, "curvature_thr": 0.05},
    "jelly_gum"  : {"type": "vinyl",       "planarity_w": 0.6, "normal_w": 0.3, "centroid_w": 0.1, "curvature_thr": 0.05},
    "tin_case"   : {"type": "metal",       "planarity_w": 0.3, "normal_w": 0.3, "centroid_w": 0.4, "curvature_thr": 0.08},
    "bottle"     : {"type": "transparent", "planarity_w": 0.4, "normal_w": 0.4, "centroid_w": 0.2, "curvature_thr": 0.03},
    "pencil_case": {"type": "flat",        "planarity_w": 0.5, "normal_w": 0.4, "centroid_w": 0.1, "curvature_thr": 0.04},
    "unknown"    : {"type": "default",     "planarity_w": 0.4, "normal_w": 0.4, "centroid_w": 0.2, "curvature_thr": 0.08},
}

def get_strategy(label: str) -> dict:
    return LABEL_STRATEGY.get(label, LABEL_STRATEGY["unknown"])


# ──────────────────────────────────────────────────────────────────────────────
# 유틸 함수
# ──────────────────────────────────────────────────────────────────────────────
def infer_unit_scale(points: np.ndarray) -> float:
    """depth 단위 자동 감지 (mm / m)"""
    z = points[:, 2]
    z = z[np.isfinite(z) & (z > 0)]
    if len(z) == 0:
        return 1.0
    if np.median(z) < 10:
        print("  [단위] meter 감지 → threshold 자동 변환")
        return 0.001
    return 1.0


def normal_to_quaternion(normal: np.ndarray) -> np.ndarray:
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    if normal[2] > 0:
        normal = -normal
    z_ref      = np.array([0.0, 0.0, -1.0])
    cross      = np.cross(z_ref, normal)
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis  = cross / cross_norm
    angle = np.arccos(np.clip(np.dot(z_ref, normal), -1.0, 1.0))
    return np.array([
        axis[0] * np.sin(angle / 2),
        axis[1] * np.sin(angle / 2),
        axis[2] * np.sin(angle / 2),
        np.cos(angle / 2),
    ])


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q
    rx = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
    ry = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1.0, 1.0))
    rz = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
    return np.degrees([rx, ry, rz])


def compute_curvature(normals: np.ndarray, knn: int = 10) -> np.ndarray:
    """법선 분산으로 곡률 추정 (낮을수록 평평함)"""
    n = len(normals)
    curvatures = np.zeros(n)
    if n < knn:
        return curvatures
    for i in range(n):
        start = max(0, i - knn // 2)
        end   = min(n, i + knn // 2)
        neighbor_normals = normals[start:end]
        mean_n = neighbor_normals.mean(axis=0)
        mean_n /= (np.linalg.norm(mean_n) + 1e-8)
        dots = np.clip([np.dot(nn, mean_n) for nn in neighbor_normals], -1, 1)
        curvatures[i] = 1.0 - np.mean(dots)
    return curvatures


# ──────────────────────────────────────────────────────────────────────────────
# SuctionEstimator 클래스
# ──────────────────────────────────────────────────────────────────────────────
class SuctionEstimator:
    """
    ROI + SAM 마스크 기반 파지점 추정기.

    사용:
        estimator = SuctionEstimator(cam)
        results = estimator.process(depth_img, rgb_img, roi, detections)

    detections 형식:
        [
            {
                "label": "haribo",
                "bbox": {"x1":100, "y1":50, "x2":200, "y2":150},
                "mask": np.ndarray (H, W) bool  ← SAM 마스크
            },
            ...
        ]
    """

    def __init__(self, cam: dict, cfg: dict = None):
        """
        Args:
            cam: {"fx", "fy", "cx", "cy"}
            cfg: picking_config.yaml (없으면 자동 로드)
        """
        self.cam = cam
        self.cfg = cfg if cfg is not None else _load_config()
        print(f"✅ SuctionEstimator 초기화 완료")
        print(f"   카메라: fx={cam['fx']}, fy={cam['fy']}, "
              f"cx={cam['cx']}, cy={cam['cy']}")

    # ── ROI depth → Point Cloud 역투영 ───────────────────────────────────────
    def _roi_depth_to_pcd(self,
                           depth_img : np.ndarray,
                           roi       : tuple,
                           rgb_img   : Optional[np.ndarray] = None
                           ) -> Tuple[o3d.geometry.PointCloud, tuple]:
        """
        ROI 영역의 depth만 역투영 → Point Cloud

        핵심: ROI 픽셀 좌표를 원본 이미지 좌표로 유지
              (카메라 파라미터는 원본 이미지 기준이므로)

        Args:
            depth_img : 전체 depth 이미지 (H, W)
            roi       : (rx1, ry1, rx2, ry2) 원본 이미지 기준
            rgb_img   : 전체 RGB 이미지 (optional)

        Returns:
            pcd     : ROI 영역 Point Cloud
            roi     : 사용된 ROI 좌표
        """
        rx1, ry1, rx2, ry2 = roi
        h_full, w_full = depth_img.shape[:2]

        # ROI 클램핑
        rx1 = max(0, rx1); ry1 = max(0, ry1)
        rx2 = min(w_full, rx2); ry2 = min(h_full, ry2)

        # ROI depth 크롭
        roi_depth = depth_img[ry1:ry2, rx1:rx2]
        roi_h, roi_w = roi_depth.shape[:2]

        # ROI 내 픽셀 그리드 (원본 이미지 좌표 기준으로 생성)
        # 중요: u, v는 원본 이미지 좌표여야 카메라 파라미터와 맞음
        u_local, v_local = np.meshgrid(
            np.arange(roi_w), np.arange(roi_h))
        u_orig = u_local + rx1   # 원본 이미지 u 좌표
        v_orig = v_local + ry1   # 원본 이미지 v 좌표

        u_flat = u_orig.flatten().astype(np.float64)
        v_flat = v_orig.flatten().astype(np.float64)
        Z_flat = roi_depth.flatten().astype(np.float64)

        # 유효 depth 필터
        valid = (Z_flat > 0) & np.isfinite(Z_flat)
        u_v   = u_flat[valid]
        v_v   = v_flat[valid]
        Z_v   = Z_flat[valid]

        if len(Z_v) == 0:
            print(f"  ⚠️  ROI depth 유효 포인트 없음")
            return o3d.geometry.PointCloud(), roi

        # 역투영 수식
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        X_v = (u_v - self.cam["cx"]) * Z_v / self.cam["fx"]
        Y_v = (v_v - self.cam["cy"]) * Z_v / self.cam["fy"]

        points = np.stack([X_v, Y_v, Z_v], axis=1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # RGB 추가 (옵션)
        if rgb_img is not None:
            roi_rgb = rgb_img[ry1:ry2, rx1:rx2]
            rgb_flat = roi_rgb.reshape(-1, 3)[valid]
            colors = rgb_flat[:, ::-1].astype(np.float64) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        print(f"  [역투영] ROI({rx1},{ry1},{rx2},{ry2}) "
              f"→ {len(points)}개 포인트")
        return pcd, (rx1, ry1, rx2, ry2)

    # ── 마스크로 물체 포인트 추출 ─────────────────────────────────────────────
    def _extract_masked_points(self,
                                pcd   : o3d.geometry.PointCloud,
                                mask  : np.ndarray,
                                roi   : tuple
                                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        SAM 마스크에 해당하는 포인트만 추출.

        Args:
            pcd  : ROI Point Cloud
            mask : (H_full, W_full) bool 마스크 (원본 이미지 기준)
            roi  : (rx1, ry1, rx2, ry2)

        Returns:
            obj_points : (N, 3) 물체 포인트
            obj_pixels : (N, 2) 픽셀 좌표 (원본 기준)
        """
        rx1, ry1, rx2, ry2 = roi
        points = np.asarray(pcd.points)

        if len(points) == 0:
            return np.empty((0, 3)), np.empty((0, 2))

        # 포인트 → 픽셀 역투영
        Z  = points[:, 2]
        u  = points[:, 0] * self.cam["fx"] / (Z + 1e-8) + self.cam["cx"]
        v  = points[:, 1] * self.cam["fy"] / (Z + 1e-8) + self.cam["cy"]

        ui = np.round(u).astype(int)
        vi = np.round(v).astype(int)

        h_mask, w_mask = mask.shape[:2]

        # 마스크 범위 내 포인트
        in_img   = (ui >= 0) & (ui < w_mask) & (vi >= 0) & (vi < h_mask)
        mask_hit = np.zeros(len(points), dtype=bool)
        mask_hit[in_img] = mask[vi[in_img], ui[in_img]]

        obj_points = points[mask_hit]
        obj_pixels = np.stack([u[mask_hit], v[mask_hit]], axis=1)

        return obj_points, obj_pixels

    # ── 노이즈 필터링 ─────────────────────────────────────────────────────────
    def _filter_noise(self,
                      points     : np.ndarray,
                      label      : str,
                      nb_neighbors: int = 20,
                      std_ratio  : float = 2.0
                      ) -> np.ndarray:
        """금속/투명 물체는 더 강한 필터링"""
        strategy = get_strategy(label)
        if strategy["type"] in ("metal", "transparent"):
            nb_neighbors = 15
            std_ratio    = 1.5

        if len(points) < nb_neighbors:
            return points

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        filtered, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio)

        removed = len(points) - len(filtered.points)
        if removed > 0:
            print(f"  [필터] 노이즈 {removed}개 제거")
        return np.asarray(filtered.points)

    # ── 투명 물체 전처리 ──────────────────────────────────────────────────────
    def _preprocess_transparent(self, points: np.ndarray) -> np.ndarray:
        """물병: z 상위 30% (카메라에 가까운 뚜껑 영역)만 사용"""
        z_thr    = np.percentile(points[:, 2], 30)
        filtered = points[points[:, 2] <= z_thr]
        print(f"  [Bottle] 뚜껑 집중: {len(filtered)}/{len(points)}개")
        return filtered

    # ── 파지점 계산 (핵심) ────────────────────────────────────────────────────
    def _compute_suction_legacy(self,
                          obj_points : np.ndarray,
                          obj_pixels : np.ndarray,
                          label      : str
                          ) -> dict:
        """
        RANSAC + 법선 + 곡률로 파지점 계산
        """
        strategy = get_strategy(label)

        # 투명 물체 전처리
        if strategy["type"] == "transparent" and len(obj_points) > 20:
            obj_points = self._preprocess_transparent(obj_points)
            if len(obj_pixels) > len(obj_points):
                obj_pixels = obj_pixels[:len(obj_points)]

        if len(obj_points) < 10:
            raise ValueError(f"포인트 부족: {len(obj_points)}개")

        # 파라미터 로드
        ransac_params = self.cfg.get("ransac_params", {})
        params        = ransac_params.get(label, self.cfg.get("default_params", {
            "distance_threshold": 5.0, "knn": 30
        }))
        knn = params["knn"]

        # 단위 감지
        unit_scale  = infer_unit_scale(obj_points)
        dist_thresh = params["distance_threshold"] * unit_scale

        # RANSAC 평면 피팅
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_points)

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=dist_thresh,
            ransac_n=3,
            num_iterations=1000,
        )
        a, b, c, d   = plane_model
        plane_normal = np.array([a, b, c])
        plane_normal /= (np.linalg.norm(plane_normal) + 1e-8)

        inlier_pts  = obj_points[inliers]
        inlier_pix  = obj_pixels[inliers] if len(obj_pixels) > 0 else None
        inlier_ratio = len(inliers) / len(obj_points)

        print(f"  [RANSAC] inlier: {len(inliers)}/{len(obj_points)} "
              f"({inlier_ratio:.2f})")

        if len(inlier_pts) < 5:
            raise ValueError("RANSAC inlier 부족")

        # ✅ 안전 모드: Open3D KNN 법선 추정 제거
        # 현재 테스트에서 estimate_normals 이후 core dump 발생 가능성이 높아서
        # RANSAC plane normal을 모든 inlier의 법선으로 사용함

        if plane_normal[2] > 0:
            plane_normal = -plane_normal

        normals = np.tile(plane_normal, (len(inlier_pts), 1))

        # RANSAC 평면 법선을 그대로 쓰므로 일관성은 1로 처리
        consistency = np.ones(len(inlier_pts), dtype=np.float32)

        # 곡률 계산도 테스트 단계에서는 생략
        curvatures = np.zeros(len(inlier_pts), dtype=np.float32)

        flat_mask = np.ones(len(inlier_pts), dtype=bool)

        # centroid 거리
        centroid   = inlier_pts.mean(axis=0)
        dists      = np.linalg.norm(inlier_pts - centroid, axis=1)
        dists_norm = dists / (dists.max() + 1e-8)

        # 최종 점수
        pw = strategy["planarity_w"]
        nw = strategy["normal_w"]
        cw = strategy["centroid_w"]

        scores = (
            nw * consistency
            + pw * inlier_ratio
            - cw * dists_norm
            - 0.2 * curvatures
        )
        scores[~flat_mask] = -np.inf

        best_idx    = np.argmax(scores)
        best_point  = inlier_pts[best_idx]
        best_normal = normals[best_idx]
        best_score  = float(scores[best_idx])

        pixel_result = (
            [int(round(inlier_pix[best_idx][0])),
             int(round(inlier_pix[best_idx][1]))]
            if inlier_pix is not None else None
        )

        quaternion = normal_to_quaternion(best_normal)
        euler_deg  = quaternion_to_euler(quaternion)

        print(f"  [파지점] ({best_point[0]:.1f}, {best_point[1]:.1f}, "
              f"{best_point[2]:.1f})  score={best_score:.3f}")
        print(f"  [법선]   ({best_normal[0]:.3f}, {best_normal[1]:.3f}, "
              f"{best_normal[2]:.3f})")
        print(f"  [오일러] {[f'{v:.1f}' for v in euler_deg]} deg")

        return {
            "position_cam" : best_point.tolist(),
            "normal"       : best_normal.tolist(),
            "quaternion"   : quaternion.tolist(),
            "euler_deg"    : euler_deg.tolist(),
            "suction_score": best_score,
            "pixel"        : pixel_result,
            "inlier_ratio" : float(inlier_ratio),
            "curvature"    : float(curvatures[best_idx]),
            "method"       : "rule_based",
            "label"        : label,
            "strategy"     : strategy["type"],
        }

    # ── 시각화 ────────────────────────────────────────────────────────────────
    def _compute_suction(self,
                          obj_points : np.ndarray,
                          obj_pixels : np.ndarray,
                          label      : str
                          ) -> dict:
        """
        RANSAC plane + centroid + PCA major axis based suction estimation.
        The public return keys are kept unchanged for process() and
        estimate_suction_rule() compatibility.
        """
        strategy = get_strategy(label)

        def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(v)
            if not np.isfinite(n) or n < 1e-8:
                return fallback.astype(np.float64)
            return (v / n).astype(np.float64)

        def _quat_from_matrix(rot: np.ndarray) -> np.ndarray:
            """Convert a 3x3 rotation matrix to [x, y, z, w]."""
            trace = np.trace(rot)
            if trace > 0.0:
                s = np.sqrt(trace + 1.0) * 2.0
                qw = 0.25 * s
                qx = (rot[2, 1] - rot[1, 2]) / s
                qy = (rot[0, 2] - rot[2, 0]) / s
                qz = (rot[1, 0] - rot[0, 1]) / s
            else:
                idx = int(np.argmax(np.diag(rot)))
                if idx == 0:
                    s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
                    qw = (rot[2, 1] - rot[1, 2]) / s
                    qx = 0.25 * s
                    qy = (rot[0, 1] + rot[1, 0]) / s
                    qz = (rot[0, 2] + rot[2, 0]) / s
                elif idx == 1:
                    s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
                    qw = (rot[0, 2] - rot[2, 0]) / s
                    qx = (rot[0, 1] + rot[1, 0]) / s
                    qy = 0.25 * s
                    qz = (rot[1, 2] + rot[2, 1]) / s
                else:
                    s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
                    qw = (rot[1, 0] - rot[0, 1]) / s
                    qx = (rot[0, 2] + rot[2, 0]) / s
                    qy = (rot[1, 2] + rot[2, 1]) / s
                    qz = 0.25 * s
            quat = np.array([qx, qy, qz, qw], dtype=np.float64)
            return _normalize(quat, np.array([0.0, 0.0, 0.0, 1.0]))

        def _axis_horizontal_score(axis: np.ndarray,
                                   center: np.ndarray) -> Tuple[float, float]:
            """
            Score how well a 3D axis matches the image bbox horizontal axis.

            _compute_suction() does not receive bbox directly, so tin_case uses
            the projected pixel direction of each PCA eigenvector. A high score
            means the gripper two points will be laid out left-right in the
            object bbox even when the case is lying sideways.
            """
            axis = _normalize(axis, np.array([1.0, 0.0, 0.0]))
            span = max(float(np.linalg.norm(np.ptp(obj_points, axis=0))) * 0.05, 1.0)
            p1 = center - axis * span
            p2 = center + axis * span
            if p1[2] <= 0 or p2[2] <= 0:
                return 0.0, 0.0
            u1 = p1[0] * self.cam["fx"] / (p1[2] + 1e-8) + self.cam["cx"]
            v1 = p1[1] * self.cam["fy"] / (p1[2] + 1e-8) + self.cam["cy"]
            u2 = p2[0] * self.cam["fx"] / (p2[2] + 1e-8) + self.cam["cx"]
            v2 = p2[1] * self.cam["fy"] / (p2[2] + 1e-8) + self.cam["cy"]
            du = float(u2 - u1)
            dv = float(v2 - v1)
            score = abs(du) / (np.hypot(du, dv) + 1e-8)
            return score, du

        is_tin_case = label == "tin_case"

        # Transparent-object preprocessing: keep pixels aligned with points.
        # For bottles, never use the cap area. First keep the visible surface
        # points, then remove the top Z band so every following step uses only
        # body points.
        if strategy["type"] == "transparent" and len(obj_points) > 20:
            z_thr = np.percentile(obj_points[:, 2], 30)
            keep = obj_points[:, 2] <= z_thr
            obj_points = obj_points[keep]
            if len(obj_pixels) == len(keep):
                obj_pixels = obj_pixels[keep]
            surface_count = len(obj_points)

            # Z top filtering: remove the upper 15% Z values from bottle
            # candidates to exclude the cap before centroid/PCA/curvature.
            cap_cut_ratio = 0.15
            body_z_max = np.percentile(obj_points[:, 2], 100.0 * (1.0 - cap_cut_ratio))
            body_keep = obj_points[:, 2] <= body_z_max
            obj_points = obj_points[body_keep]
            if len(obj_pixels) == len(body_keep):
                obj_pixels = obj_pixels[body_keep]

            print(f"  [Transparent] surface focus: {surface_count}/{len(keep)} points")
            print(f"  [Bottle body] removed top Z {cap_cut_ratio*100:.0f}% "
                  f"(z>{body_z_max:.3f}); body points={len(obj_points)}")

        # Tin case body filtering: remove top/bottom Z bands so lid/body seam
        # and edge transition points do not become suction candidates. The
        # centroid, PCA axis, gripper points, and curvature below all use this
        # filtered body point cloud.
        if is_tin_case and len(obj_points) > 30:
            body_cut_ratio = 0.10
            z_low = np.percentile(obj_points[:, 2], 100.0 * body_cut_ratio)
            z_high = np.percentile(obj_points[:, 2], 100.0 * (1.0 - body_cut_ratio))
            body_keep = (obj_points[:, 2] >= z_low) & (obj_points[:, 2] <= z_high)
            if int(body_keep.sum()) >= 10:
                before_count = len(obj_points)
                obj_points = obj_points[body_keep]
                if len(obj_pixels) == len(body_keep):
                    obj_pixels = obj_pixels[body_keep]
                print(f"  [Tin body] removed Z bottom/top {body_cut_ratio*100:.0f}% "
                      f"({z_low:.3f}..{z_high:.3f}); "
                      f"body points={len(obj_points)}/{before_count}")

        if len(obj_points) < 10:
            raise ValueError(f"Not enough object points: {len(obj_points)}")

        # 1) Plane estimation: fit the suction surface with RANSAC.
        ransac_params = self.cfg.get("ransac_params", {})
        params = ransac_params.get(label, self.cfg.get("default_params", {
            "distance_threshold": 5.0, "knn": 30
        }))
        unit_scale = infer_unit_scale(obj_points)
        dist_thresh = params["distance_threshold"] * unit_scale

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_points)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=dist_thresh,
            ransac_n=3,
            num_iterations=1000,
        )
        a, b, c, d = plane_model
        plane_normal = _normalize(np.array([a, b, c], dtype=np.float64),
                                  np.array([0.0, 0.0, -1.0]))
        if plane_normal[2] > 0:
            plane_normal = -plane_normal
            d = -d

        inlier_pts = obj_points[inliers]
        inlier_pix = obj_pixels[inliers] if len(obj_pixels) == len(obj_points) else None
        inlier_ratio = len(inliers) / len(obj_points)
        print(f"  [RANSAC] inlier: {len(inliers)}/{len(obj_points)} "
              f"({inlier_ratio:.2f})")

        if len(inlier_pts) < 5:
            raise ValueError("Not enough RANSAC inliers")

        # 2) Centroid: use the object/body center as the suction point.
        # For bottles this is computed after cap removal, so the suction point
        # stays on the body instead of the cap.
        centroid = inlier_pts.mean(axis=0)

        # 3) PCA major axis and gripper placement: project body inliers to the
        # plane, align gripper X with the long axis, then place two gripper
        # points around the body centroid.
        centered = inlier_pts - centroid
        projected = centered - np.outer(centered @ plane_normal, plane_normal)
        cov = np.cov(projected, rowvar=False) if len(projected) >= 3 else np.eye(3)
        eigvals, eigvecs = np.linalg.eigh(cov)
        if is_tin_case:
            # Side-lying tin_case handling: choose the PCA eigenvector whose
            # projected image direction best matches the bbox horizontal axis.
            # This keeps gripper points arranged along the case's long, wide
            # direction instead of accidentally selecting a vertical/edge axis.
            candidate_order = np.argsort(eigvals)[::-1][:2]
            best_axis = None
            best_score = -np.inf
            best_du = 0.0
            for eig_idx in candidate_order:
                axis = _normalize(eigvecs[:, int(eig_idx)], np.array([1.0, 0.0, 0.0]))
                axis = axis - plane_normal * np.dot(axis, plane_normal)
                axis_norm = np.linalg.norm(axis)
                if axis_norm < 1e-6:
                    continue
                axis = axis / axis_norm
                score, du = _axis_horizontal_score(axis, centroid)
                if score > best_score:
                    best_axis = axis
                    best_score = score
                    best_du = du
            if best_axis is None:
                fallback_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                fallback_axis = fallback_axis - plane_normal * np.dot(fallback_axis, plane_normal)
                best_axis = _normalize(fallback_axis, np.array([1.0, 0.0, 0.0]))
                best_score, best_du = _axis_horizontal_score(best_axis, centroid)
            major_axis = best_axis

            # 180-degree flip: keep the chosen tin_case axis left-to-right in
            # image space so gripper point ordering is stable.
            if best_du < 0:
                major_axis = -major_axis
            print(f"  [Tin PCA] horizontal alignment={best_score:.3f}")
        else:
            major_axis = _normalize(eigvecs[:, int(np.argmax(eigvals))],
                                    np.array([1.0, 0.0, 0.0]))
            major_axis = _normalize(
                major_axis - plane_normal * np.dot(major_axis, plane_normal),
                np.array([1.0, 0.0, 0.0])
            )
            if major_axis[0] < 0:
                major_axis = -major_axis

        axis_coords = projected @ major_axis
        half_span = max((float(axis_coords.max()) - float(axis_coords.min())) * 0.5,
                        dist_thresh)
        gripper_p1 = centroid - major_axis * half_span
        gripper_p2 = centroid + major_axis * half_span

        # Pose orientation: local -Z follows the plane normal and local X
        # follows the PCA long axis, making the suction gripper parallel to it.
        z_axis = _normalize(-plane_normal, np.array([0.0, 0.0, 1.0]))
        y_axis = _normalize(np.cross(z_axis, major_axis), np.array([0.0, 1.0, 0.0]))
        x_axis = _normalize(np.cross(y_axis, z_axis), major_axis)
        rot = np.column_stack([x_axis, y_axis, z_axis])

        # 4) Curvature/flatness scoring.
        # Bottle/transparent objects and tin_case body points use real local
        # curvature from k-NN PCA:
        #   curvature = smallest eigenvalue / sum(eigenvalues)
        # Flatter local patches get higher curvature_scores; highly curved
        # bottle surfaces or tin_case lid/body seams are penalized.
        if strategy["type"] == "transparent" or is_tin_case:
            curvature_knn = int(params.get("knn", 30))
            curvature_knn = max(5, min(curvature_knn, len(inlier_pts)))
            curvature_thr = float(strategy.get("curvature_thr", 0.03))
            curvatures = np.zeros(len(inlier_pts), dtype=np.float64)

            inlier_pcd = o3d.geometry.PointCloud()
            inlier_pcd.points = o3d.utility.Vector3dVector(inlier_pts)
            kdtree = o3d.geometry.KDTreeFlann(inlier_pcd)

            for idx, point in enumerate(inlier_pts):
                _, nn_idx, _ = kdtree.search_knn_vector_3d(point, curvature_knn)
                local_pts = inlier_pts[nn_idx]
                local_centered = local_pts - local_pts.mean(axis=0)
                local_cov = np.cov(local_centered, rowvar=False)
                eig = np.clip(np.linalg.eigvalsh(local_cov), 0.0, None)
                eig_sum = float(eig.sum())
                curvatures[idx] = float(eig[0] / (eig_sum + 1e-8))

            curvature_scores = 1.0 - np.clip(
                curvatures / (curvature_thr + 1e-8), 0.0, 1.0)
            curvature_method = "knn_pca_body" if is_tin_case else "knn_pca"
        else:
            # Flat, vinyl, and unknown objects keep the previous plane residual
            # flatness score from the RANSAC plane.
            plane_residual = np.abs(inlier_pts @ plane_normal + d)
            curvature_scores = 1.0 - np.clip(
                plane_residual / (dist_thresh + 1e-8), 0.0, 1.0)
            curvatures = 1.0 - curvature_scores
            curvature_method = "plane_residual"

        # 5) Candidate scoring: normal alignment, center proximity, flatness,
        # and ROI-safe area. Final selected position remains the centroid.

        radial = np.linalg.norm(projected, axis=1)
        center_scores = 1.0 - radial / (radial.max() + 1e-8)

        normal_alignment = abs(float(np.dot(plane_normal, np.array([0.0, 0.0, -1.0]))))
        normal_scores = np.full(len(inlier_pts), normal_alignment, dtype=np.float64)

        roi_scores = np.ones(len(inlier_pts), dtype=np.float64)
        if inlier_pix is not None and len(inlier_pix) == len(inlier_pts):
            min_uv = inlier_pix.min(axis=0)
            max_uv = inlier_pix.max(axis=0)
            span_uv = np.maximum(max_uv - min_uv, 1.0)
            margin = span_uv * 0.10
            safe_min = min_uv + margin
            safe_max = max_uv - margin
            safe = np.all((inlier_pix >= safe_min) & (inlier_pix <= safe_max), axis=1)
            edge_dist = np.minimum(inlier_pix - min_uv, max_uv - inlier_pix)
            edge_score = np.clip(np.min(edge_dist / (margin + 1e-8), axis=1), 0.0, 1.0)
            roi_scores = np.where(safe, 1.0, edge_score)

        pw = strategy["planarity_w"]
        nw = strategy["normal_w"]
        cw = strategy["centroid_w"]
        rw = 0.15
        scores = (
            nw * normal_scores
            + cw * center_scores
            + pw * curvature_scores
            + rw * roi_scores
        ) / (nw + cw + pw + rw + 1e-8)

        best_idx = int(np.argmax(scores))
        best_point = centroid
        best_normal = plane_normal
        best_score = float(scores[best_idx] * (0.5 + 0.5 * inlier_ratio))

        pixel_result = None
        if centroid[2] > 0:
            pixel_result = [
                int(round(centroid[0] * self.cam["fx"] / (centroid[2] + 1e-8) + self.cam["cx"])),
                int(round(centroid[1] * self.cam["fy"] / (centroid[2] + 1e-8) + self.cam["cy"])),
            ]
        elif inlier_pix is not None:
            pixel_result = [
                int(round(np.mean(inlier_pix[:, 0]))),
                int(round(np.mean(inlier_pix[:, 1]))),
            ]

        quaternion = _quat_from_matrix(rot)
        euler_deg = quaternion_to_euler(quaternion)

        print(f"  [Suction]  ({best_point[0]:.1f}, {best_point[1]:.1f}, "
              f"{best_point[2]:.1f})  score={best_score:.3f}")
        print(f"  [Euler]    {[f'{v:.1f}' for v in euler_deg]} deg")

        return {
            "position_cam" : best_point.tolist(),
            "normal"       : best_normal.tolist(),
            "quaternion"   : quaternion.tolist(),
            "euler_deg"    : euler_deg.tolist(),
            "suction_score": best_score,
            "pixel"        : pixel_result,
            "inlier_ratio" : float(inlier_ratio),
            "curvature"    : float(curvatures[best_idx]),
            "method"       : "rule_based",
            "label"        : label,
            "strategy"     : strategy["type"],
        }

    def visualize(self,
                  rgb_img    : np.ndarray,
                  results    : list,
                  save_path  : Optional[str] = None) -> np.ndarray:
        """
        파지점 시각화 (RGB 이미지에 파지점 + 법선 방향 표시)
        """
        vis = rgb_img.copy()

        for r in results:
            pixel = r.get("pixel")
            if pixel is None:
                continue

            px, py   = int(pixel[0]), int(pixel[1])
            label    = r.get("label", "")
            score    = r.get("suction_score", 0)
            strategy = r.get("strategy", "")
            normal   = r.get("normal", [0, 0, 1])

            # 파지점 원
            cv2.circle(vis, (px, py), 12, (0, 255, 0), -1)
            cv2.circle(vis, (px, py), 14, (255, 255, 255), 2)

            # 법선 화살표
            nx = int(normal[0] * 30)
            ny = int(normal[1] * 30)
            cv2.arrowedLine(vis, (px, py), (px+nx, py+ny),
                            (0, 0, 255), 2, tipLength=0.3)

            # 텍스트
            text = f"{label}[{strategy}] {score:.2f}"
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis,
                          (px-5, py-th-20), (px+tw+5, py-5),
                          (0, 0, 0), -1)
            cv2.putText(vis, text, (px, py-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if save_path:
            cv2.imwrite(save_path, vis)
            print(f"  [시각화] 저장: {save_path}")

        return vis

    # ── 메인 처리 함수 ────────────────────────────────────────────────────────
    def process(self,
                depth_img  : np.ndarray,
                rgb_img    : np.ndarray,
                roi        : tuple,
                detections : list) -> list:
        """
        ROI + 탐지 결과로 모든 물체의 파지점 계산.

        Args:
            depth_img  : 전체 depth 이미지 (H, W)
            rgb_img    : 전체 RGB 이미지 (H, W, 3)
            roi        : (rx1, ry1, rx2, ry2) 바구니 ROI
            detections : DINO+SAM 탐지 결과 리스트
                         [{"label", "bbox", "mask", ...}, ...]

        Returns:
            results: 각 물체별 파지점 정보 리스트
                     [{"label", "position_cam", "normal",
                       "quaternion", "euler_deg", "pixel", ...}, ...]
        """
        print(f"\n{'='*55}")
        print(f"[SuctionEstimator] ROI: {roi}, 탐지 물체: {len(detections)}개")

        # 1. ROI depth → Point Cloud 역투영
        roi_pcd, roi_used = self._roi_depth_to_pcd(
            depth_img, roi, rgb_img=rgb_img)

        if len(roi_pcd.points) == 0:
            print("  ❌ ROI Point Cloud 생성 실패")
            return []

        results = []

        # 2. 탐지된 물체별 처리
        for i, det in enumerate(detections):
            label = det.get("label", "unknown")
            mask  = det.get("mask")   # SAM 마스크 (H_full, W_full) bool
            bbox  = det.get("bbox")   # {"x1", "y1", "x2", "y2"}

            print(f"\n  [{i+1}] {label}")

            try:
                # 3. 마스크로 물체 포인트 추출
                if mask is not None:
                    obj_points, obj_pixels = self._extract_masked_points(
                        roi_pcd, mask, roi_used)
                    print(f"  [마스크] 추출 포인트: {len(obj_points)}개")
                else:
                    # 마스크 없으면 bbox로 fallback
                    print("  [마스크 없음] bbox fallback 사용")
                    bbox_tuple = (
                        bbox["x1"], bbox["y1"],
                        bbox["x2"], bbox["y2"])
                    obj_points, obj_pixels = self._extract_masked_points(
                        roi_pcd,
                        self._bbox_to_mask(bbox_tuple, depth_img.shape),
                        roi_used)

                # 포인트 부족 시 skip
                if len(obj_points) < 10:
                    print(f"  ⚠️  포인트 부족 ({len(obj_points)}개) → skip")
                    continue

                # 4. 노이즈 필터링
                obj_points = self._filter_noise(obj_points, label)
                if len(obj_points) < 10:
                    print(f"  ⚠️  필터링 후 포인트 부족 → skip")
                    continue

                # 5. 파지점 계산
                result = self._compute_suction(obj_points, obj_pixels, label)

                # 탐지 정보 추가
                result["detection_id"] = det.get("id", i + 1)
                result["confidence"]   = det.get("confidence", 0.0)
                if bbox:
                    cx = (bbox["x1"] + bbox["x2"]) // 2
                    cy = (bbox["y1"] + bbox["y2"]) // 2
                    result["center_pixel"] = {"x": cx, "y": cy}

                results.append(result)

            except Exception as e:
                print(f"  ❌ {label} 파지점 계산 실패: {e}")
                continue

        print(f"\n[완료] {len(results)}/{len(detections)}개 파지점 계산 성공")
        return results

    # ── bbox → 마스크 변환 (fallback용) ──────────────────────────────────────
    @staticmethod
    def _bbox_to_mask(bbox: tuple, img_shape: tuple) -> np.ndarray:
        """bbox를 마스크로 변환 (마스크 없을 때 fallback)"""
        h, w = img_shape[:2]
        x1, y1, x2, y2 = bbox
        mask = np.zeros((h, w), dtype=bool)
        mask[max(0,y1):min(h,y2), max(0,x1):min(w,x2)] = True
        return mask


# ──────────────────────────────────────────────────────────────────────────────
# 기존 인터페이스 호환 (pick_n_place.py에서 그대로 호출 가능)
# ──────────────────────────────────────────────────────────────────────────────
def estimate_suction_rule(pcd   : o3d.geometry.PointCloud,
                           bbox  : tuple,
                           cam   : dict,
                           label : str = "unknown",
                           mask  : Optional[np.ndarray] = None,
                           cfg   : Optional[dict] = None) -> dict:
    """
    기존 코드와의 호환성 유지용 함수.
    새 코드에서는 SuctionEstimator.process() 사용 권장.
    """
    if cfg is None:
        cfg = _load_config()

    estimator = SuctionEstimator(cam, cfg)
    points    = np.asarray(pcd.points)

    valid        = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    valid_points = points[valid]
    Z = valid_points[:, 2]
    u = valid_points[:, 0] * cam["fx"] / (Z + 1e-8) + cam["cx"]
    v = valid_points[:, 1] * cam["fy"] / (Z + 1e-8) + cam["cy"]
    pixels = np.stack([u, v], axis=1)

    x1, y1, x2, y2 = bbox
    in_bbox = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)

    if mask is not None:
        h, w = mask.shape[:2]
        ui = np.round(u).astype(int)
        vi = np.round(v).astype(int)
        in_img   = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        mask_hit = np.zeros(len(in_bbox), dtype=bool)
        mask_hit[in_img] = mask[vi[in_img], ui[in_img]] > 0
        selected = in_bbox & mask_hit
    else:
        selected = in_bbox

    obj_points = valid_points[selected]
    obj_pixels = pixels[selected]

    if len(obj_points) < 10 and mask is not None:
        selected   = in_bbox
        obj_points = valid_points[selected]
        obj_pixels = pixels[selected]

    obj_points = estimator._filter_noise(obj_points, label)
    return estimator._compute_suction(obj_points, obj_pixels, label)


def compute_pick_priority(results: list) -> list:
    """
    Sort process() results into pick/fallback order.

    The original result keys are preserved. Returned items are shallow copies
    with priority metadata added, so callers can try rank 1 first and move to
    the next item automatically if a pick attempt fails.
    """
    if not results:
        return []

    def _safe_float(value, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            value = float(value)
            return value if np.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    def _position_z(result: dict) -> float:
        # Step 1: use camera/object Z as box height priority.
        position = result.get("position_cam", [0.0, 0.0, 0.0])
        if isinstance(position, dict):
            return _safe_float(position.get("z"), 0.0)
        if len(position) >= 3:
            return _safe_float(position[2], 0.0)
        return 0.0

    def _confidence_factor(result: dict) -> float:
        # Step 2: optionally reflect detection/mask confidence when available.
        # Missing confidence defaults to 1.0 so legacy callers are unaffected.
        factor = 1.0
        for key in ("mask_confidence", "mask_score", "segmentation_score",
                    "confidence", "detection_confidence"):
            conf = result.get(key)
            if conf is None:
                continue
            conf = _safe_float(conf, 1.0)
            if conf > 0.0:
                factor *= conf
        return factor

    prioritized = []
    for original_idx, result in enumerate(results):
        candidate = dict(result)

        z_value = _position_z(candidate)
        suction_score = _safe_float(candidate.get("suction_score"), 0.0)
        confidence_factor = _confidence_factor(candidate)

        # Step 3: compute the score used after Z priority. Confidence is
        # optional and only affects results when confidence keys exist.
        priority_score = suction_score * confidence_factor

        candidate["_priority_z"] = z_value
        candidate["_priority_original_idx"] = original_idx
        candidate["priority_score"] = float(priority_score)
        prioritized.append(candidate)

    # Step 4: final ordering. Highest Z wins, ties prefer higher score, then
    # raw suction score, then the original order.
    prioritized.sort(
        key=lambda item: (
            item["_priority_z"],
            item["priority_score"],
            _safe_float(item.get("suction_score"), 0.0),
            -item["_priority_original_idx"],
        ),
        reverse=True,
    )

    fallback_order = [
        item.get("detection_id", item.get("label", idx + 1))
        for idx, item in enumerate(prioritized)
    ]

    # Step 5: add fallback metadata. If rank 1 fails, caller should try the
    # next rank in this returned list.
    for rank, item in enumerate(prioritized, start=1):
        item["pick_priority"] = rank
        item["fallback_rank"] = rank
        item["fallback_order"] = fallback_order
        item.pop("_priority_z", None)
        item.pop("_priority_original_idx", None)

    return prioritized


# Local PLY/JSON test helpers and CLI entrypoint removed for production.
