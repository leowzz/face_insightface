"""Video frame sampling utilities."""

from __future__ import annotations

from collections.abc import Iterator

import cv2
from loguru import logger

from .schemas import FramePacket


def iter_sampled_frames(
    video_path: str,
    sample_fps: float,
    start_time_sec: float = 0.0,
    max_duration_sec: float | None = None,
) -> Iterator[FramePacket]:
    """Yield sampled frames from video.

    :param video_path: 视频路径。
    :param sample_fps: 每秒采样帧数。
    :param start_time_sec: 起始处理时间（秒）。
    :param max_duration_sec: 最大处理时长（秒），为空表示全视频。
    :yield: 抽样帧对象。
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if native_fps <= 0:
        native_fps = 25.0

    if start_time_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000.0)

    sample_interval = max(int(round(native_fps / sample_fps)), 1)
    logger.info(f"{video_path=}, {native_fps=:.3f}, {sample_fps=}, {start_time_sec=}, {sample_interval=}")

    local_index = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        local_index += 1
        if local_index % sample_interval != 0:
            continue

        frame_index = max(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1, 0)
        frame_time_sec = frame_index / native_fps
        elapsed_from_start = max(frame_time_sec - start_time_sec, 0.0)
        if max_duration_sec is not None and elapsed_from_start >= max_duration_sec:
            logger.info(
                f"达到最大处理时长，停止抽帧: {max_duration_sec=}, {start_time_sec=}, {frame_time_sec=:.3f}"
            )
            break

        yield FramePacket(
            frame_index=frame_index,
            frame_time_sec=frame_time_sec,
            frame_bgr=frame,
        )

    cap.release()
