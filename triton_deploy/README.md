# Triton 高性能人脸识别部署

基于 NVIDIA Triton Inference Server 的 SCRFD + ArcFace 人脸检测与识别服务。

## 目录结构

```
triton_deploy/
├── model_repository/
│   ├── det_10g/
│   │   ├── config.pbtxt
│   │   └── 1/model.onnx          ← 由 prepare_models.sh 复制
│   └── w600k_r50/
│       ├── config.pbtxt
│       └── 1/model.onnx          ← 由 prepare_models.sh 复制
├── client/
│   ├── face_client.py            ← 主客户端（完整推理流程）
│   ├── preprocess.py             ← letterbox + 归一化 + 人脸对齐
│   └── postprocess.py            ← SCRFD anchor decode + NMS
├── docker-compose.yml
├── prepare_models.sh
├── requirements-client.txt
└── README.md
```

## 快速开始

### Step 1：准备模型

```bash
cd triton_deploy
bash prepare_models.sh
```

脚本会从 `~/.insightface/models/buffalo_l/` 复制模型，并打印真实的 input/output tensor 名称。
请对照输出核实 `config.pbtxt` 中的字段（开发阶段 `strict-model-config=false` 可自动推断，无需手动修改）。

### Step 2：启动 Triton

```bash
docker compose up -d
docker compose logs -f   # 等待出现 "Started GRPCInferenceService"
```

### Step 3：健康检查

```bash
curl http://localhost:8000/v2/health/ready          # → {"ready": true}
curl http://localhost:8000/v2/models/det_10g         # 确认检测模型已加载
curl http://localhost:8000/v2/models/w600k_r50       # 确认识别模型已加载
```

### Step 4：运行客户端

```bash
pip install -r requirements-client.txt
cd client
python face_client.py ../../data/leo01.png
# 期望输出：检测到 1 张人脸，embedding_norm ≈ 1.0
```

## 关键技术决策

| 决策 | 选择 | 原因 |
|------|------|------|
| batch 模式 | `max_batch_size: 0` | buffalo_l ONNX 导出时 batch 固定为 1 |
| Triton 镜像 | `tritonserver:24.08-py3-cpu` | 无 GPU 依赖，体积小 |
| strict-model-config | `false` | SCRFD 输出节点名是数字，关闭后自动推断 |
| 客户端协议 | `tritonclient.http` | 简单易调试，curl 可直接验证 |
| 预处理位置 | 客户端 Python | 灵活调试，letterbox resize <1ms 不是瓶颈 |

## 端口说明

| 端口 | 协议 | 用途 |
|------|------|------|
| 8000 | HTTP | REST API / 客户端推理 |
| 8001 | gRPC | 高性能推理（可选） |
| 8002 | HTTP | Prometheus metrics |

## API 说明（Python）

```python
from client.face_client import FaceTritonClient

client = FaceTritonClient(url="localhost:8000")

# 完整流程：检测 + 识别
faces = client.get_faces(img_bgr)
# faces[i] = {"bbox": [x1,y1,x2,y2], "score": float, "kps": [5,2], "embedding": [512]}

# 仅检测
bboxes, scores, kps = client.detect(img_bgr)

# 仅识别（需要关键点）
embedding = client.get_embedding(img_bgr, kps[0])
```
