"""
pick_n_place.py
───────────────────────────────────────────────────────────
PickNPlace 추론 파이프라인
 
grpc_mode 가 이 클래스를 다음 4개 API 로만 호출한다.
 
    1) PickNPlace(logger, config, weight, options, cuda)   - 생성자
    2) set_intrinsic(cx, cy, fx, fy)                       - 카메라 intrinsic 적용
    3) run(rgb, depth, normal, ...)                        - 추론 메인
    4) save_result(rgb, predictions, polygons, ...)        - 시각화 결과 저장
 
수정사항:
  - A1. save_result 시각화 좌표 역변환 (로봇 좌표 → 카메라 좌표 → 픽셀)
  - A2. state 판정 수정 (더미 포인트 state=1 방지, suction_score 기준)
  - A3. save_result try/except 가드 (시각화 실패가 응답 죽이지 않도록)
  - B2. valid_dets 정렬 추가
  - ranking 연결 (configs/ranking_config.yaml 기반)
  - depth_image → estimate_suction_rule 전달 (tilt 회전용)
"""
 
from typing import Any, Dict, List, Optional, Tuple
import logging
import os
 
import cv2
import numpy as np
import open3d as o3d
import yaml
 
from src.detector.dino_sam import load_models, run_detection
from src.classifier.ensemble_classifier import EnsembleClassifier
from src.pipeline.rule_based_suction import estimate_suction_rule
from src.pipeline.suction_ranking import rank_suction_candidates
from scipy.spatial.transform import Rotation    #[수정]
 
class PickNPlace:
    def __init__(
            self,
            logger: logging.Logger,
            config_name: str,
            checkpoint: str,
            options: Dict[str, Any],
            cuda: str = "0",
    ) -> None:
        self.logger      = logger
        self.name        = "PickNPlace"
        self.options     = options
        self.config_name = config_name
        self.checkpoint  = checkpoint
        self.cuda        = cuda
 
        # ── 경로 ──────────────────────────────────────────────────────────────
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.picknplace_dir = os.path.abspath(os.path.join(base_dir, ".."))
 
        # ── config 로드 ───────────────────────────────────────────────────────
        self.det_cfg     = self._load_yaml("configs/detector_config.yaml")
        self.picking_cfg = self._load_yaml("configs/picking_config.yaml")
 
        # ── intrinsic 초기화 ──────────────────────────────────────────────────
        self.cam      = {"cx": 0.0, "cy": 0.0, "fx": 0.0, "fy": 0.0}
        self.c_matrix = np.eye(3, dtype=np.float64)
 
        # ── extrinsic 로드 ────────────────────────────────────────────────────
        extrinsic_path = os.path.join(self.picknplace_dir, "configs", "extrinsic.npy")
        if os.path.exists(extrinsic_path):
            self.extrinsic = np.load(extrinsic_path)
            self.logger.info(f"[{self.name}] extrinsic 로드: {extrinsic_path}")
        else:
            self.extrinsic = np.eye(4, dtype=np.float64)
            self.logger.warning(f"[{self.name}] extrinsic 없음 → identity 사용")
 
        # ── extrinsic 역행렬 (시각화용) ───────────────────────────────────────
        try:
            self.extrinsic_inv = np.linalg.inv(self.extrinsic)
        except Exception:
            self.extrinsic_inv = np.eye(4, dtype=np.float64)
 
        # ── DINO + SAM 로드 ───────────────────────────────────────────────────
        self.logger.info(f"[{self.name}] DINO + SAM2 로드 중...")
        self.processor, self.gd_model, self.sam_predictor = load_models(self.det_cfg)
        self.logger.info(f"[{self.name}] DINO + SAM2 로드 완료")
 
        # ── 분류기 로드 ───────────────────────────────────────────────────────
        self.logger.info(f"[{self.name}] EnsembleClassifier 로드 중...")
        self.classifier = EnsembleClassifier()
        self.logger.info(f"[{self.name}] EnsembleClassifier 로드 완료")
 
        self.logger.info(f"[{self.name}] 초기화 완료")
 
    # ─── 유틸 ─────────────────────────────────────────────────────────────────
 
    def _load_yaml(self, rel_path: str) -> dict:
        full_path = os.path.join(self.picknplace_dir, rel_path)
        with open(full_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f.read())
 
    # ─── 카메라 intrinsic ─────────────────────────────────────────────────────
 
    def set_intrinsic(self, cx: float, cy: float, fx: float, fy: float) -> None:
        """카메라 intrinsic 4개 파라미터를 받아 내부 c_matrix 에 반영."""
        self.cam = {"cx": cx, "cy": cy, "fx": fx, "fy": fy}
        self.c_matrix = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1],
        ], dtype=np.float64)
        self.logger.info(
            f"[{self.name}] intrinsic 갱신: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
 
    # ─── depth → point cloud ──────────────────────────────────────────────────
 
    def _depth_to_pcd(self, depth: np.ndarray) -> o3d.geometry.PointCloud:
        """depth 이미지 → open3d PointCloud (카메라 좌표계, mm 단위)"""
        h, w   = depth.shape
        fx, fy = self.cam["fx"], self.cam["fy"]
        cx, cy = self.cam["cx"], self.cam["cy"]
 
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.astype(np.float64)
        x = (us - cx) * z / (fx + 1e-8)
        y = (vs - cy) * z / (fy + 1e-8)
 
        pts   = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
        pts   = pts[valid]
 
        pcd        = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd
 
    # ─── 카메라 → 로봇 좌표 변환 ─────────────────────────────────────────────
 
    # 수정 후

    def _cam_to_robot(self, xyz_cam: np.ndarray, quat_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """카메라 3D 좌표 + 방향 → 로봇 베이스 좌표 (extrinsic 적용)"""
        R_cam = Rotation.from_quat(quat_cam).as_matrix()

        T_cam = np.eye(4, dtype=np.float64)
        T_cam[:3, :3] = R_cam
        T_cam[:3,  3] = xyz_cam

        T_robot = self.extrinsic @ T_cam

        xyz_robot  = T_robot[:3, 3]
        quat_robot = Rotation.from_matrix(T_robot[:3, :3]).as_quat()
        return xyz_robot, quat_robot
 
    # ─── 추론 메인 ────────────────────────────────────────────────────────────
 
    def run(
            self,
            rgb_image           : np.ndarray,
            depth_image         : Optional[np.ndarray] = None,
            normal_image        : Optional[np.ndarray] = None,
            depth_scale         : float = 1000,
            compute_suction_pts : bool = False,
            vis_pcd             : bool = False,
            save_ply            : bool = False,
            roi_2d              : Optional[List[float]] = None,
    ) -> Tuple[Dict[str, Any], List[Any]]:
        """
        추론 메인. 반환은 (result, predictions) 튜플.
 
        파이프라인:
          1) DINO + SAM → bbox, mask
          2) EnsembleClassifier → class_id, label
          3) depth → PointCloud
          4) rule_based_suction → suction point (카메라 좌표)
          5) extrinsic → 로봇 베이스 좌표 변환
          6) rank_suction_candidates → priority_score 기반 정렬
          7) result dict 조립
        """
        self.logger.info(f"[{self.name}] run() 호출")
        self.logger.info(f"[{self.name}]   rgb   : {rgb_image.shape if rgb_image is not None else None}")
        self.logger.info(f"[{self.name}]   depth : {depth_image.shape if depth_image is not None else None}")
        self.logger.info(f"[{self.name}]   roi_2d: {roi_2d}")
 
        # roi 문자열 → int 변환
        if roi_2d:
            roi_2d = [int(x) for x in roi_2d]
 
        try:
            # ── 1) DINO + SAM 탐지 ────────────────────────────────────────────
            _, det_results, masks = run_detection(
                rgb_image,
                self.processor,
                self.gd_model,
                self.sam_predictor,
                roi=roi_2d,
                cfg=self.det_cfg,
            )
 
            if not det_results:
                self.logger.info(f"[{self.name}] 객체 미검출 → state=-2")
                return self._empty_result(roi_2d, state=-2), []
 
            # ── 2) 분류 + polygon 추출 ────────────────────────────────────────
            label_to_id = {
                "bottle"     : 0,
                "haribo"     : 1,
                "jelly_gum"  : 2,
                "pencil_case": 3,
                "tin_case"   : 4,
                "Unknown"    : 5,
            }
 
            polygons    = []
            class_ids   = []
            labels      = []
            valid_masks = []
            valid_dets  = []
 
            for i, det in enumerate(det_results):
                bbox = (det["bbox"]["x1"], det["bbox"]["y1"],
                        det["bbox"]["x2"], det["bbox"]["y2"])
 
                x1, y1, x2, y2 = [int(v) for v in bbox]
                crop = rgb_image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
 
                label, score, method = self.classifier.predict(crop)
                self.logger.info(
                    f"[{self.name}] 객체 {i+1}: {label} ({score:.3f}, {method})")
 
                mask = masks[i] if i < len(masks) else None
                if mask is not None:
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8),
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # 가장 큰 contour 선택 (파편 방지)
                        contour = max(contours, key=cv2.contourArea)
                        polygon = contour.squeeze().tolist()
                        if isinstance(polygon[0], int):
                            polygon = [polygon]
                    else:
                        polygon = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                else:
                    polygon = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
 
                polygons.append(polygon)
                class_ids.append(label_to_id.get(label, 5))
                labels.append(label)
                valid_masks.append(mask)
                valid_dets.append(det)
 
            if not polygons:
                return self._empty_result(roi_2d, state=-1), []
 
            # ── 3) depth → PointCloud ─────────────────────────────────────────
            pcd = None
            if depth_image is not None:
                pcd = self._depth_to_pcd(depth_image)
 
            # ── 4~5) suction point 계산 ───────────────────────────────────────
            suction_results = []
            suction_points  = []
            suction_scores  = []
 
            for i, det in enumerate(valid_dets):
                bbox  = (det["bbox"]["x1"], det["bbox"]["y1"],
                         det["bbox"]["x2"], det["bbox"]["y2"])
                label = labels[i]
                mask  = valid_masks[i]
 
                if pcd is None or len(np.asarray(pcd.points)) == 0:
                    # depth 없으면 더미 포인트
                    suction_points.append([
                        ([det["center"]["x"], det["center"]["y"], 0.0],
                         [0.0, 0.0, 0.0, 1.0])
                    ])
                    suction_scores.append(-1.0)
                    suction_results.append({
                        "detection_id" : det.get("id", i + 1),
                        "label"        : label,
                        "position_cam" : [det["center"]["x"],
                                          det["center"]["y"], 0.0],
                        "suction_score": -1.0,
                        "pixel"        : None,
                        "mask"         : mask,
                    })
                    continue
 
                try:
                    result = estimate_suction_rule(
                        pcd, bbox, self.cam,
                        label=label,
                        mask=mask,
                        cfg=self.picking_cfg,
                        depth_img=depth_image,   # tilt 회전용
                    )
 
                    # 카메라 → 로봇 좌표 변환
                    xyz_robot, quat = self._cam_to_robot(
                        result["position_cam"],
                        result["quaternion"],
                    )
 
                    suction_points.append([(xyz_robot.tolist(), quat)])
                    suction_scores.append(float(result.get("suction_score", 0.0)))
 
                    # ranking용 메타데이터 추가
                    result["detection_id"] = det.get("id", i + 1)
                    result["confidence"]   = det.get("confidence", 0.0)
                    result["mask"]         = mask
                    suction_results.append(result)
 
                    self.logger.info(
                        f"[{self.name}] 객체 {i+1} suction: {xyz_robot.tolist()}")
 
                except Exception as e:
                    self.logger.warning(
                        f"[{self.name}] 객체 {i+1} suction 실패: {e}")
                    suction_points.append([
                        ([det["center"]["x"], det["center"]["y"], 0.0],
                         [0.0, 0.0, 0.0, 1.0])
                    ])
                    suction_scores.append(-1.0)
                    suction_results.append({
                        "detection_id" : det.get("id", i + 1),
                        "label"        : label,
                        "position_cam" : [det["center"]["x"],
                                          det["center"]["y"], 0.0],
                        "suction_score": -1.0,
                        "pixel"        : None,
                        "mask"         : mask,
                    })
 
            # ── 6) ranking → priority_score 기반 정렬 ────────────────────────
            try:
                ranked_results = rank_suction_candidates(
                    suction_results,
                    detections=valid_dets,
                    depth=depth_image,
                )
                id_to_rank = {
                    r.get("detection_id"): r.get("priority_rank", 999)
                    for r in ranked_results
                }
                order = sorted(
                    range(len(valid_dets)),
                    key=lambda idx: id_to_rank.get(
                        valid_dets[idx].get("id", idx + 1), 999)
                )
                polygons       = [polygons[idx]       for idx in order]
                class_ids      = [class_ids[idx]      for idx in order]
                suction_points = [suction_points[idx] for idx in order]
                suction_scores = [suction_scores[idx] for idx in order]
                valid_dets     = [valid_dets[idx]     for idx in order]
 
                self.logger.info(
                    f"[{self.name}] ranking 완료: "
                    f"{[round(r.get('priority_score', 0), 3) for r in ranked_results]}")
 
            except Exception as e:
                # ranking 실패해도 기존 순서 유지
                self.logger.warning(
                    f"[{self.name}] ranking 실패 (기존 순서 유지): {e}")
 
            # A2: suction_score > 0 인 것이 하나라도 있어야 state=1
            state = 1 if any(s > 0 for s in suction_scores) else -1
 
            result = {
                "result_data"   : polygons,
                "state"         : state,
                "class_id"      : class_ids,
                "pts_per_object": [len(sp) for sp in suction_points],
                "suction_points": suction_points,
                "2d_roi"        : roi_2d if roi_2d else [],
                "3d_roi"        : [],
            }
            return result, valid_dets
 
        except Exception as e:
            self.logger.error(f"[{self.name}] run() 예외: {e}")
            return self._empty_result(roi_2d, state=0), []
 
    # ─── 빈 결과 ──────────────────────────────────────────────────────────────
 
    def _empty_result(self, roi_2d, state: int) -> Dict[str, Any]:
        return {
            "result_data"   : [],
            "state"         : state,
            "class_id"      : [],
            "pts_per_object": [],
            "suction_points": [],
            "2d_roi"        : roi_2d if roi_2d else [],
            "3d_roi"        : [],
        }
 
    # ─── 결과 시각화 저장 ─────────────────────────────────────────────────────
 
    def save_result(
            self,
            rgb_img            : np.ndarray,
            predictions        : Any,
            polygons           : List[Any],
            suction_pts        : Optional[Any] = None,
            save_path          : str = ".",
            save_name          : str = "temp.png",
            compute_suction_pts: bool = False,
    ) -> None:
        """추론 결과를 이미지로 시각화해서 저장."""
 
        # A3: save_result 예외가 추론 응답에 영향을 주지 않도록
        try:
            if not save_path:
                self.logger.warning(
                    f"[{self.name}] save_result: save_path 없음 → 저장 생략")
                return
 
            os.makedirs(save_path, exist_ok=True)
            full_path = os.path.join(save_path, f"{save_name}.png")
 
            vis_img = rgb_img.copy()
            h, w    = vis_img.shape[:2]
 
            # polygon 오버레이
            for polygon in polygons:
                pts = np.array(polygon, dtype=np.int32)
                if pts.ndim == 1:
                    pts = pts.reshape(-1, 1, 2)
                cv2.polylines(vis_img, [pts], True, (0, 255, 0), 2)
 
            # A1: suction point 오버레이
            # suction_pts의 xyz는 로봇 좌표 → extrinsic 역변환 후 카메라 좌표로 투영
            if compute_suction_pts and suction_pts and self.cam["fx"] > 0:
                for obj_pts in suction_pts:
                    for pt in obj_pts:
                        xyz, _ = pt
 
                        # 로봇 좌표 → 카메라 좌표
                        p_robot = np.array([xyz[0], xyz[1], xyz[2], 1.0])
                        p_cam   = (self.extrinsic_inv @ p_robot)[:3]
                        xc, yc, zc = p_cam
 
                        # z <= 0: 더미 포인트 또는 카메라 뒤쪽 → 스킵
                        if zc <= 0:
                            continue
 
                        # 카메라 좌표 → 픽셀
                        u = int(xc * self.cam["fx"] / zc + self.cam["cx"])
                        v = int(yc * self.cam["fy"] / zc + self.cam["cy"])
 
                        if 0 <= u < w and 0 <= v < h:
                            cv2.circle(vis_img, (u, v), 8, (0, 0, 255), -1)
                            cv2.circle(vis_img, (u, v), 10, (255, 255, 255), 2)
 
            cv2.imwrite(full_path, vis_img)
            self.logger.info(f"[{self.name}] save_result: {full_path}")
 
        except Exception as e:
            # A3: 시각화 실패는 로그만 남기고 응답 무관
            self.logger.warning(
                f"[{self.name}] save_result 실패 (응답 무관): {e}")
 