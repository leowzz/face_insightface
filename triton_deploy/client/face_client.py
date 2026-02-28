"""
face_client.py - Triton Inference Server 人脸检测+识别客户端

用法:
    python face_client.py <image_path>
    python face_client.py ../../data/leo01.png
"""
import sys
import time
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
from loguru import logger

from preprocess import prepare_det_input, align_face, prepare_rec_input
from postprocess import decode_scrfd_outputs, rescale_to_original


class FaceTritonClient:
    def __init__(
        self,
        url: str = "localhost:8000",
        det_model: str = "det_10g",
        rec_model: str = "w600k_r50",
        det_size: tuple = (640, 640),
        conf_thresh: float = 0.5,
        iou_thresh: float = 0.45,
    ):
        logger.info("连接 Triton Server: {}", url)
        self.client = httpclient.InferenceServerClient(url=url)
        self.det_model = det_model
        self.rec_model = rec_model
        self.det_size = det_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        self._det_output_names = self._get_output_names(det_model)
        logger.debug("det_10g outputs: {}", self._det_output_names)

        self._rec_output_names = self._get_output_names(rec_model)
        logger.debug("w600k_r50 outputs: {}", self._rec_output_names)

        logger.info("模型元数据加载完成，conf_thresh={}, iou_thresh={}", conf_thresh, iou_thresh)

    def _get_output_names(self, model_name: str) -> list:
        meta = self.client.get_model_metadata(model_name)
        return [o["name"] for o in meta["outputs"]]

    def _infer(self, model_name: str, input_name: str, data: np.ndarray, output_names: list) -> dict:
        inputs = [httpclient.InferInput(input_name, data.shape, np_to_triton_dtype(data.dtype))]
        inputs[0].set_data_from_numpy(data)
        outputs = [httpclient.InferRequestedOutput(name) for name in output_names]

        t0 = time.perf_counter()
        result = self.client.infer(model_name, inputs, outputs=outputs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("[{}] 推理耗时 {:.1f} ms", model_name, elapsed_ms)

        return {name: result.as_numpy(name) for name in output_names}

    def detect(self, img_bgr: np.ndarray) -> tuple:
        """
        检测图像中的人脸。
        返回: (bboxes [N,4], scores [N], kps [N,5,2])  — 原图坐标
        """
        h, w = img_bgr.shape[:2]
        logger.debug("开始检测，输入图像尺寸 {}x{}", w, h)

        blob, scale, dw, dh = prepare_det_input(img_bgr, self.det_size)
        logger.debug("letterbox 预处理完成，scale={:.4f}, pad=(dw={:.1f}, dh={:.1f})", scale, dw, dh)

        outputs = self._infer(self.det_model, "input.1", blob, self._det_output_names)

        bboxes, scores, kps = decode_scrfd_outputs(
            outputs,
            input_h=self.det_size[0],
            input_w=self.det_size[1],
            conf_thresh=self.conf_thresh,
            iou_thresh=self.iou_thresh,
        )
        bboxes, kps = rescale_to_original(bboxes, kps, scale, dw, dh)

        logger.info("检测完成，发现 {} 张人脸（conf>={}, iou<={}）", len(bboxes), self.conf_thresh, self.iou_thresh)
        for i, (bbox, score) in enumerate(zip(bboxes, scores)):
            logger.debug(
                "  人脸 {}: bbox=[{:.0f},{:.0f},{:.0f},{:.0f}] score={:.4f}",
                i + 1, bbox[0], bbox[1], bbox[2], bbox[3], score,
            )

        return bboxes, scores, kps

    def get_embedding(self, img_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """
        对单张人脸做对齐 + 识别，返回 512-dim L2 归一化 embedding。
        kps: [5, 2] 原图关键点坐标
        """
        logger.debug("开始人脸对齐")
        aligned = align_face(img_bgr, kps)

        blob = prepare_rec_input(aligned)
        outputs = self._infer(self.rec_model, "input.1", blob, self._rec_output_names)

        emb = list(outputs.values())[0].flatten()
        norm_before = float(np.linalg.norm(emb))
        emb = emb / (norm_before + 1e-6)
        logger.debug("识别完成，原始 norm={:.4f}，归一化后 norm={:.6f}", norm_before, np.linalg.norm(emb))

        return emb

    def get_faces(self, img_bgr: np.ndarray) -> list:
        """
        完整流程：检测 + 识别所有人脸。
        返回字典列表，每个字典包含:
            bbox      [4]    x1,y1,x2,y2
            score     float
            kps       [5,2]
            embedding [512]
        """
        t0 = time.perf_counter()
        bboxes, scores, kps_all = self.detect(img_bgr)

        faces = []
        for i, (bbox, score, kps) in enumerate(zip(bboxes, scores, kps_all)):
            logger.debug("处理第 {}/{} 张人脸的 embedding", i + 1, len(bboxes))
            emb = self.get_embedding(img_bgr, kps)
            faces.append({
                "bbox": bbox,
                "score": float(score),
                "kps": kps,
                "embedding": emb,
            })

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("全流程完成：{} 张人脸，总耗时 {:.1f} ms", len(faces), elapsed_ms)
        return faces


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import cv2

    logger.remove()
    logger.add(sys.stderr, level="DEBUG", colorize=True,
               format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}")

    if len(sys.argv) < 2:
        logger.error("用法: python face_client.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    logger.info("读取图片: {}", img_path)
    img = cv2.imread(img_path)
    if img is None:
        logger.error("无法读取图片: {}", img_path)
        sys.exit(1)

    client = FaceTritonClient()
    faces = client.get_faces(img)

    logger.info("========== 结果汇总 ==========")
    for i, face in enumerate(faces):
        bbox = face["bbox"]
        emb = face["embedding"]
        logger.info(
            "人脸 {}: bbox=[{:.0f},{:.0f},{:.0f},{:.0f}] score={:.4f} emb_norm={:.6f}",
            i + 1, bbox[0], bbox[1], bbox[2], bbox[3], face["score"], np.linalg.norm(emb),
        )
