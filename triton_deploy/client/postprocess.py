"""
postprocess.py - SCRFD 检测模型输出的 anchor decode + NMS
"""
import numpy as np


def generate_anchors(feat_h: int, feat_w: int, stride: int) -> np.ndarray:
    """生成 SCRFD anchor 中心坐标，每个位置 2 个 anchor。"""
    cy = np.arange(feat_h) * stride
    cx = np.arange(feat_w) * stride
    cx, cy = np.meshgrid(cx, cy)
    # 每个位置重复 2 次（num_anchors=2）
    centers = np.stack([cx, cy], axis=-1).reshape(-1, 2)
    centers = np.repeat(centers, 2, axis=0)
    return centers.astype(np.float32)


def _iou(box, boxes):
    """计算单个 box 与 boxes 的 IoU。"""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_box + area_boxes - inter + 1e-6)


def nms(bboxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45) -> np.ndarray:
    """贪心 NMS，返回保留的索引。"""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = _iou(bboxes[i], bboxes[order[1:]])
        order = order[1:][ious < iou_thresh]
    return np.array(keep, dtype=np.int64)


def decode_scrfd_outputs(
    outputs: dict,
    input_h: int = 640,
    input_w: int = 640,
    conf_thresh: float = 0.5,
    iou_thresh: float = 0.45,
) -> tuple:
    """
    解码 SCRFD 三个 stride (8/16/32) 的输出并做 NMS。

    outputs: {tensor_name: numpy_array} — 从 Triton 返回的字典
    返回: (bboxes [N,4], scores [N], kps [N,5,2])  均为原始输入尺寸坐标
    """
    strides = [8, 16, 32]
    all_bboxes, all_scores, all_kps = [], [], []

    # 优先按已知 tensor 名取值，避免依赖字典顺序导致错位解码
    score_names = ["448", "471", "494"]
    bbox_names = ["451", "474", "497"]
    kps_names = ["454", "477", "500"]
    has_named_outputs = all(n in outputs for n in (score_names + bbox_names + kps_names))

    if has_named_outputs:
        score_tensors = [outputs[n] for n in score_names]
        bbox_tensors = [outputs[n] for n in bbox_names]
        kps_tensors = [outputs[n] for n in kps_names]
    else:
        # 兼容旧逻辑：按输出顺序回退
        tensors = list(outputs.values())
        score_tensors = tensors[0:3]
        bbox_tensors = tensors[3:6]
        kps_tensors = tensors[6:9]

    for i, stride in enumerate(strides):
        score_raw = score_tensors[i].reshape(-1, 1)   # [N_anchors, 1]
        bbox_raw = bbox_tensors[i].reshape(-1, 4)     # [N_anchors, 4]
        kps_raw = kps_tensors[i].reshape(-1, 10)      # [N_anchors, 10]

        # 有些导出的 SCRFD score 已是概率 [0,1]，不能重复 sigmoid
        smin = float(score_raw.min())
        smax = float(score_raw.max())
        if 0.0 <= smin and smax <= 1.0:
            scores = score_raw.flatten()
        else:
            scores = 1.0 / (1.0 + np.exp(-score_raw)).flatten()

        mask = scores >= conf_thresh
        if not mask.any():
            continue

        scores = scores[mask]
        bbox_raw = bbox_raw[mask]
        kps_raw = kps_raw[mask]

        feat_h = input_h // stride
        feat_w = input_w // stride
        centers = generate_anchors(feat_h, feat_w, stride)[mask]

        # ltrb → xyxy
        x1 = centers[:, 0] - bbox_raw[:, 0] * stride
        y1 = centers[:, 1] - bbox_raw[:, 1] * stride
        x2 = centers[:, 0] + bbox_raw[:, 2] * stride
        y2 = centers[:, 1] + bbox_raw[:, 3] * stride
        bboxes = np.stack([x1, y1, x2, y2], axis=1)

        # keypoints
        kps = kps_raw.reshape(-1, 5, 2) * stride
        kps[:, :, 0] += centers[:, 0:1]
        kps[:, :, 1] += centers[:, 1:2]

        all_bboxes.append(bboxes)
        all_scores.append(scores)
        all_kps.append(kps)

    if not all_bboxes:
        return np.empty((0, 4)), np.empty(0), np.empty((0, 5, 2))

    all_bboxes = np.concatenate(all_bboxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_kps    = np.concatenate(all_kps, axis=0)

    keep = nms(all_bboxes, all_scores, iou_thresh)
    return all_bboxes[keep], all_scores[keep], all_kps[keep]


def rescale_to_original(
    bboxes: np.ndarray,
    kps: np.ndarray,
    scale: float,
    dw: float,
    dh: float,
) -> tuple:
    """将 letterbox 坐标映射回原图坐标。"""
    if bboxes.shape[0] == 0:
        return bboxes, kps

    bboxes = bboxes.copy()
    bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - dw) / scale
    bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - dh) / scale

    kps = kps.copy()
    kps[:, :, 0] = (kps[:, :, 0] - dw) / scale
    kps[:, :, 1] = (kps[:, :, 1] - dh) / scale

    return bboxes, kps
