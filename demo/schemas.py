"""Demo schema definitions (pydantic v2)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class DemoConfig(BaseModel):
    """CLI 配置参数。"""

    video_path: Path
    triton_url: str = Field(default="localhost:8000")
    sample_fps: float = Field(default=1.0, gt=0)
    start_time_sec: float = Field(default=0.0, ge=0)
    max_duration_sec: Optional[float] = Field(default=None, gt=0)
    output_dir: Path = Field(default=Path("demo/output"))
    similarity_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    det_conf_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    min_face_size: int = Field(default=80, ge=1)
    blur_var_threshold: float = Field(default=80.0, ge=0.0)
    max_pose_yaw_dev: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_pose_roll_deg: Optional[float] = Field(default=None, ge=0.0, le=90.0)


class FramePacket(BaseModel):
    """抽帧结果。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_index: int = Field(ge=0)
    frame_time_sec: float = Field(ge=0)
    frame_bgr: np.ndarray


class DetectedFace(BaseModel):
    """单张人脸检测识别结果。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bbox: np.ndarray
    score: float
    kps: np.ndarray | None = None
    age: float | None = None
    gender: int | None = None
    pose: np.ndarray | None = None
    embedding: np.ndarray


class VideoRow(BaseModel):
    """videos 表记录。"""

    id: int
    path: str
    fps_used: float
    duration_limit_sec: Optional[float] = None


class FaceInsert(BaseModel):
    """写入 faces 表的数据。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    video_id: int
    frame_index: int = Field(ge=0)
    frame_time_sec: float = Field(ge=0)
    score: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    blur_var: float
    bbox_w: float
    bbox_h: float
    has_kps: int
    pose_yaw: float | None = None
    pose_pitch: float | None = None
    pose_roll: float | None = None
    age: float | None = None
    gender: int | None = None
    crop_path: str
    embedding: np.ndarray


class FaceEmbeddingRow(BaseModel):
    """用于聚类的人脸向量。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    face_id: int
    embedding: np.ndarray


class ClusterAssignment(BaseModel):
    """聚类回写结果。"""

    face_id: int
    cluster_id: int = Field(ge=1)


class ClusterView(BaseModel):
    """页面展示所需聚类信息。"""

    cluster_id: int
    face_count: int
    preview_paths: list[str]
    avg_score: float
    avg_blur_var: float
    avg_bbox_w: float
    avg_bbox_h: float
    avg_age: float | None = None
    dominant_gender: str | None = None
