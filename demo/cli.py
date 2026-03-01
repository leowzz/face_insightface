"""Movie face clustering demo CLI."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from time import perf_counter
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from .cluster import greedy_cluster
from .db import (
    commit,
    connect_db,
    count_faces,
    init_db,
    insert_face,
    insert_video,
    load_cluster_views,
    load_face_embeddings,
    save_cluster_assignments,
)
from .frame_extract import iter_sampled_frames
from .html import render_html, write_html
from .schemas import DemoConfig, DetectedFace, FaceInsert


def _setup_perf_logger(run_dir: Path) -> tuple:
    perf_log_path = run_dir / "performance.log"
    sink_id = logger.add(
        perf_log_path,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
        filter=lambda record: record["extra"].get("perf", False),
    )
    return logger.bind(perf=True), sink_id, perf_log_path


def _parse_hms_to_seconds(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"start-time 格式错误，应为 h:m:s，当前为: {value}")
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    except ValueError as exc:
        raise ValueError(f"start-time 含有非法数字: {value}") from exc

    if hours < 0 or minutes < 0 or seconds < 0:
        raise ValueError(f"start-time 不能为负数: {value}")
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"start-time 的 m/s 必须小于 60: {value}")
    return hours * 3600 + minutes * 60 + seconds


def _load_face_client_class() -> type:
    """Load FaceTritonClient from triton_deploy/client."""

    root = Path(__file__).resolve().parents[1]
    client_dir = root / "triton_deploy" / "client"
    if str(client_dir) not in sys.path:
        sys.path.append(str(client_dir))
    from face_client import FaceTritonClient  # pylint: disable=import-outside-toplevel

    return FaceTritonClient


class _LocalInsightFaceClient:
    """Local fallback client when Triton is unavailable."""

    def __init__(self, det_size: tuple[int, int] = (640, 640), conf_thresh: float = 0.65):
        from insightface.app import FaceAnalysis  # pylint: disable=import-outside-toplevel

        self._app = FaceAnalysis(providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=0, det_size=det_size)
        self._conf_thresh = conf_thresh

    def get_faces(self, img_bgr: np.ndarray) -> list[dict]:
        raw_faces = self._app.get(img_bgr)
        faces: list[dict] = []
        for face in raw_faces:
            score = float(getattr(face, "det_score", 0.0))
            if score < self._conf_thresh:
                continue

            embedding = np.asarray(
                getattr(face, "normed_embedding", getattr(face, "embedding", np.array([], dtype=np.float32))),
                dtype=np.float32,
            )
            emb_norm = float(np.linalg.norm(embedding))
            if emb_norm > 0:
                embedding = embedding / (emb_norm + 1e-6)

            kps = getattr(face, "kps", None)
            faces.append(
                {
                    "bbox": np.asarray(face.bbox, dtype=np.float32),
                    "score": score,
                    "kps": np.asarray(kps, dtype=np.float32) if kps is not None else None,
                    "embedding": embedding,
                }
            )
        return faces


def _build_face_client(config: DemoConfig):
    try:
        FaceTritonClient = _load_face_client_class()
        client = FaceTritonClient(
            url=config.triton_url,
            det_size=(640, 640),
            conf_thresh=config.det_conf_threshold,
        )
        logger.info("使用 Triton 客户端进行推理")
        return client
    except Exception as exc:  # noqa: BLE001
        logger.warning("Triton 不可用，切换本地 InsightFace 推理: {}", exc)
        return _LocalInsightFaceClient(det_size=(640, 640), conf_thresh=config.det_conf_threshold)


def _parse_args(argv: list[str] | None = None) -> DemoConfig:
    parser = argparse.ArgumentParser(description="电影人脸聚类 Demo")
    parser.add_argument("video_path", help="电影文件路径")
    parser.add_argument("--fps", type=float, default=1.0, help="每秒抽帧数量，默认 1")
    parser.add_argument("--start-time", default="0:0:0", help="起始时间，格式 h:m:s，例如 0:03:30")
    parser.add_argument("--duration", type=float, default=None, help="处理时长（秒），不填则处理完整视频")
    parser.add_argument("--triton-url", default=os.environ.get("TRITON_SERVER_URL", "localhost:8000"))
    parser.add_argument("--output-dir", default="demo/output")
    parser.add_argument("--similarity-threshold", type=float, default=0.65, help="聚类相似度阈值，默认 0.65")
    parser.add_argument("--det-confidence-threshold", type=float, default=0.65, help="检测置信度阈值，默认 0.65")
    parser.add_argument("--min-face-size", type=int, default=80, help="最小人脸框边长（像素），默认 80")
    parser.add_argument("--blur-threshold", type=float, default=80.0, help="清晰度阈值（Laplacian 方差），默认 80")
    parser.add_argument(
        "--max-pose-yaw-dev",
        type=float,
        default=None,
        help="可选姿态过滤：鼻尖到双眼距离不对称度上限（0~1），例如 0.35",
    )
    parser.add_argument(
        "--max-pose-roll-deg",
        type=float,
        default=None,
        help="可选姿态过滤：双眼连线倾斜角绝对值上限（度），例如 25",
    )
    args = parser.parse_args(argv)

    return DemoConfig(
        video_path=Path(args.video_path).expanduser().resolve(),
        triton_url=args.triton_url,
        sample_fps=args.fps,
        start_time_sec=_parse_hms_to_seconds(args.start_time),
        max_duration_sec=args.duration,
        output_dir=Path(args.output_dir).expanduser().resolve(),
        similarity_threshold=args.similarity_threshold,
        det_conf_threshold=args.det_confidence_threshold,
        min_face_size=args.min_face_size,
        blur_var_threshold=args.blur_threshold,
        max_pose_yaw_dev=args.max_pose_yaw_dev,
        max_pose_roll_deg=args.max_pose_roll_deg,
    )


def _normalize_faces(raw_faces: list) -> list[DetectedFace]:
    normalized: list[DetectedFace] = []
    for raw in raw_faces:
        normalized.append(
            DetectedFace(
                bbox=np.asarray(raw["bbox"], dtype=np.float32),
                score=float(raw["score"]),
                kps=np.asarray(raw["kps"], dtype=np.float32) if "kps" in raw and raw["kps"] is not None else None,
                embedding=np.asarray(raw["embedding"], dtype=np.float32),
            )
        )
    return normalized


def _crop_face(frame_bgr: np.ndarray, bbox: np.ndarray, padding_ratio: float = 0.15) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox.tolist()]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)

    padx = bw * padding_ratio
    pady = bh * padding_ratio

    sx1 = max(int(x1 - padx), 0)
    sy1 = max(int(y1 - pady), 0)
    sx2 = min(int(x2 + padx), width)
    sy2 = min(int(y2 + pady), height)

    if sx2 <= sx1 or sy2 <= sy1:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return frame_bgr[sy1:sy2, sx1:sx2]


def _laplacian_variance(image_bgr: np.ndarray) -> float:
    if image_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _pose_metrics(kps: np.ndarray | None) -> tuple[float | None, float | None]:
    if kps is None or kps.shape != (5, 2):
        return None, None

    left_eye = kps[0]
    right_eye = kps[1]
    nose = kps[2]
    eye_dx = float(right_eye[0] - left_eye[0])
    eye_dy = float(right_eye[1] - left_eye[1])
    roll_deg = abs(float(np.degrees(np.arctan2(eye_dy, eye_dx))))

    d_left = float(np.linalg.norm(nose - left_eye))
    d_right = float(np.linalg.norm(right_eye - nose))
    denom = max(d_left + d_right, 1e-6)
    yaw_dev = abs(d_left - d_right) / denom
    return yaw_dev, roll_deg


def run_demo(config: DemoConfig) -> Path:
    """Run end-to-end demo.

    :param config: Demo 配置。
    :return: 结果 HTML 路径。
    """

    if not config.video_path.exists():
        raise FileNotFoundError(f"视频不存在: {config.video_path}")

    run_id = hashlib.md5(str(config.video_path).encode("utf-8")).hexdigest()[:10]  # noqa: S324
    run_dir = config.output_dir / run_id
    faces_dir = run_dir / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)

    db_path = run_dir / "movie_faces.db"
    logger.info(f"{run_dir=}, {db_path=}")

    perf_logger, perf_sink_id, perf_log_path = _setup_perf_logger(run_dir)
    logger.info(f"性能日志输出: {perf_log_path}")
    conn = None
    total_t0 = perf_counter()
    html_path = run_dir / "index.html"
    try:
        perf_logger.info(
            f"run_start|{config.video_path=}|{config.triton_url=}|{config.sample_fps=}|"
            f"{config.start_time_sec=}|{config.max_duration_sec=}|{config.det_conf_threshold=}|"
            f"{config.min_face_size=}|{config.blur_var_threshold=}|{config.max_pose_yaw_dev=}|"
            f"{config.max_pose_roll_deg=}|{config.similarity_threshold=}"
        )

        db_init_t0 = perf_counter()
        conn = connect_db(db_path)
        init_db(conn)
        video_row = insert_video(
            conn=conn,
            path=str(config.video_path),
            fps_used=config.sample_fps,
            duration_limit_sec=config.max_duration_sec,
        )
        db_init_ms = (perf_counter() - db_init_t0) * 1000
        perf_logger.info(f"db_init_done|{db_init_ms=:.3f}|{video_row.id=}")

        client_init_t0 = perf_counter()
        client = _build_face_client(config)
        client_init_ms = (perf_counter() - client_init_t0) * 1000
        perf_logger.info(f"triton_client_init_done|{client_init_ms=:.3f}")

        frame_count = 0
        face_count = 0
        raw_detect_count = 0
        filtered_count = 0
        filtered_by_score = 0
        filtered_by_size = 0
        filtered_by_empty_crop = 0
        filtered_by_blur = 0
        filtered_by_pose = 0
        triton_total_ms = 0.0
        persist_total_ms = 0.0
        frame_loop_t0 = perf_counter()
        for packet in iter_sampled_frames(
            video_path=str(config.video_path),
            sample_fps=config.sample_fps,
            start_time_sec=config.start_time_sec,
            max_duration_sec=config.max_duration_sec,
        ):
            frame_stage_t0 = perf_counter()
            frame_count += 1

            triton_t0 = perf_counter()
            faces = _normalize_faces(client.get_faces(packet.frame_bgr))
            raw_detect_count += len(faces)
            triton_ms = (perf_counter() - triton_t0) * 1000
            triton_total_ms += triton_ms

            persist_t0 = perf_counter()
            for face_idx, face in enumerate(faces):
                if face.score < config.det_conf_threshold:
                    filtered_count += 1
                    filtered_by_score += 1
                    continue

                face_w = float(face.bbox[2] - face.bbox[0])
                face_h = float(face.bbox[3] - face.bbox[1])
                if min(face_w, face_h) < config.min_face_size:
                    filtered_count += 1
                    filtered_by_size += 1
                    continue

                crop = _crop_face(packet.frame_bgr, face.bbox)
                if crop.size == 0:
                    filtered_count += 1
                    filtered_by_empty_crop += 1
                    continue

                blur_var = _laplacian_variance(crop)
                if blur_var < config.blur_var_threshold:
                    filtered_count += 1
                    filtered_by_blur += 1
                    continue

                yaw_dev, roll_deg = _pose_metrics(face.kps)
                if config.max_pose_yaw_dev is not None and yaw_dev is not None and yaw_dev > config.max_pose_yaw_dev:
                    filtered_count += 1
                    filtered_by_pose += 1
                    continue
                if config.max_pose_roll_deg is not None and roll_deg is not None and roll_deg > config.max_pose_roll_deg:
                    filtered_count += 1
                    filtered_by_pose += 1
                    continue

                rel_crop = Path("faces") / f"frame_{packet.frame_index:08d}_{face_idx:02d}.jpg"
                abs_crop = run_dir / rel_crop
                cv2.imwrite(str(abs_crop), crop)

                insert_face(
                    conn,
                    FaceInsert(
                        video_id=video_row.id,
                        frame_index=packet.frame_index,
                        frame_time_sec=packet.frame_time_sec,
                        score=face.score,
                        bbox_x1=float(face.bbox[0]),
                        bbox_y1=float(face.bbox[1]),
                        bbox_x2=float(face.bbox[2]),
                        bbox_y2=float(face.bbox[3]),
                        crop_path=rel_crop.as_posix(),
                        embedding=face.embedding,
                    ),
                )
                face_count += 1
            persist_ms = (perf_counter() - persist_t0) * 1000
            persist_total_ms += persist_ms

            frame_total_ms = (perf_counter() - frame_stage_t0) * 1000
            perf_logger.info(
                f"frame_done|{frame_count=}|{packet.frame_index=}|{packet.frame_time_sec=:.3f}|"
                f"{len(faces)=}|{triton_ms=:.3f}|{persist_ms=:.3f}|{frame_total_ms=:.3f}"
            )

            if frame_count % 20 == 0:
                commit(conn)
                logger.info(f"处理中: {frame_count=}, {face_count=}")
                avg_triton_ms = triton_total_ms / frame_count
                avg_persist_ms = persist_total_ms / frame_count
                perf_logger.info(
                    f"progress|{frame_count=}|{face_count=}|{avg_triton_ms=:.3f}|{avg_persist_ms=:.3f}"
                )

        commit(conn)
        frame_loop_ms = (perf_counter() - frame_loop_t0) * 1000
        logger.info(f"抽帧完成: {frame_count=}, {face_count=}")
        perf_logger.info(
            f"frame_loop_done|{frame_loop_ms=:.3f}|{frame_count=}|{raw_detect_count=}|{face_count=}|"
            f"{filtered_count=}|{filtered_by_score=}|{filtered_by_size=}|{filtered_by_empty_crop=}|"
            f"{filtered_by_blur=}|{filtered_by_pose=}"
        )

        cluster_t0 = perf_counter()
        embedding_rows = load_face_embeddings(conn, video_row.id)
        assignments = greedy_cluster(
            rows=embedding_rows,
            similarity_threshold=config.similarity_threshold,
        )
        save_cluster_assignments(conn, assignments)
        cluster_ms = (perf_counter() - cluster_t0) * 1000
        perf_logger.info(f"cluster_done|{cluster_ms=:.3f}|{len(embedding_rows)=}|{len(assignments)=}")

        html_t0 = perf_counter()
        clusters = load_cluster_views(conn, video_row.id, preview_limit=12)
        total_faces = count_faces(conn, video_row.id)
        html_content = render_html(
            video_path=str(config.video_path),
            clusters=clusters,
            total_faces=total_faces,
        )
        write_html(html_path, html_content)
        html_ms = (perf_counter() - html_t0) * 1000
        perf_logger.info(f"html_done|{html_ms=:.3f}|{len(clusters)=}|{total_faces=}")

        total_ms = (perf_counter() - total_t0) * 1000
        effective_fps = frame_count / (total_ms / 1000.0) if total_ms > 0 else 0.0
        perf_logger.info(
            f"run_done|{total_ms=:.3f}|{frame_count=}|{face_count=}|"
            f"{effective_fps=:.3f}|{perf_log_path=}"
        )
        logger.info(f"处理完成: {html_path=}, {len(clusters)=}, {total_faces=}")
        logger.info(
            "质量过滤统计: raw_detect_count={} accepted_count={} filtered_count={} "
            "filtered_by_score={} filtered_by_size={} filtered_by_empty_crop={} "
            "filtered_by_blur={} filtered_by_pose={}",
            raw_detect_count,
            face_count,
            filtered_count,
            filtered_by_score,
            filtered_by_size,
            filtered_by_empty_crop,
            filtered_by_blur,
            filtered_by_pose,
        )
        return html_path
    finally:
        if conn is not None:
            conn.close()
        logger.remove(perf_sink_id)


def main(argv: list[str] | None = None) -> int:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    try:
        config = _parse_args(argv)
        html_path = run_demo(config)
        print(f"结果页面: file://{html_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"执行失败: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
