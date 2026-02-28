"""
preprocess.py - 检测模型和识别模型的预处理函数
"""
import cv2
import numpy as np


def prepare_det_input(img_bgr: np.ndarray, det_size: tuple = (640, 640)):
    """
    为 SCRFD 检测模型准备输入。
    - Letterbox resize 到 det_size，保持宽高比
    - BGR → RGB
    - 归一化到 [-1, 1]（mean=127.5, std=128）
    - 返回 (blob [1,3,H,W] float32, scale, dw, dh)
    """
    target_h, target_w = det_size
    h, w = img_bgr.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw = (target_w - new_w) / 2
    dh = (target_h - new_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    blob = ((rgb.astype(np.float32) - 127.5) / 128.0)
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]  # [1, 3, H, W]

    return blob, scale, dw, dh


def align_face(img_bgr: np.ndarray, kps: np.ndarray, output_size: int = 112) -> np.ndarray:
    """
    基于 5 点关键点做仿射变换，将人脸对齐到 112×112。
    kps: [5, 2] (x, y) 原图坐标
    """
    # ArcFace 标准 112×112 模板关键点
    dst_pts = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)

    src_pts = kps.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    aligned = cv2.warpAffine(img_bgr, M, (output_size, output_size),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return aligned


def prepare_rec_input(aligned_bgr: np.ndarray) -> np.ndarray:
    """
    为 ArcFace 识别模型准备输入。
    - BGR → RGB
    - 归一化到 [-1, 1]（mean=0.5, std=0.5，即除以 255 再 *2-1）
    - 返回 [1, 3, 112, 112] float32
    """
    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    blob = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]  # [1, 3, 112, 112]
    return blob
