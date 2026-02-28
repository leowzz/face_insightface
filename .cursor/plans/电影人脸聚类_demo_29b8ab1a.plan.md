---
name: 电影人脸聚类 Demo
overview: 实现一个 Demo：对传入的电影路径按 FPS 抽帧，用 Triton 做人脸检测与 512 维向量提取，SQLite 持久化 + 简单向量聚类，最后生成静态 HTML 页面按“人数/图片数”倒序展示电影中出现的所有人。
todos: []
isProject: false
---

# 电影人脸聚类 Demo 实现计划

## 目标

- **输入**：电影文件路径（如 `/path/to/movie.mp4`）
- **流程**：按可配置 FPS 抽帧 → 每帧调用 Triton 检测+识别 → 人脸 crop 落盘 + embedding 与元数据入 SQLite → 简单向量聚类（同片内同一人归为一类）
- **输出**：生成一个静态 HTML 页面，展示该电影中出现的“人物”（聚类），按**图片数量倒序**排列，每人可配若干缩略图

## 架构与数据流

```mermaid
flowchart LR
  Video[视频路径] --> Extract[抽帧]
  Extract --> Triton[Triton 检测+embedding]
  Triton --> Store[SQLite + 人脸 crop 文件]
  Store --> Cluster[聚类]
  Cluster --> HTML[生成静态 HTML]
  HTML --> Browser[浏览器打开]
```



## 1. 目录与依赖

- **新目录**：`demo/`，作为 demo 包（可 `python -m demo` 运行）。
- **依赖**：仅用现有栈即可完成核心逻辑；若希望聚类更稳可加 `scikit-learn`（见下）。当前计划先用 **numpy 贪心聚类**，不新增依赖。
  - 现有：[pyproject.toml](pyproject.toml) 已含 `opencv-python`、`numpy`、`tritonclient[http]`、`loguru`。
  - 可选：在 [pyproject.toml](pyproject.toml) 中增加 `scikit-learn`，用于 DBSCAN/层次聚类（备选方案）。

## 2. 抽帧

- 使用 `cv2.VideoCapture(video_path)` 按**时间间隔**抽帧（例如每秒 1 帧，可通过参数配置，如 `--fps 1`）。
- 实现方式：根据 `cap.get(cv2.CAP_PROP_FPS)` 得到视频 FPS，计算“每隔 N 帧取一帧”的 N，或按 `cap.get(cv2.CAP_PROP_POS_MSEC)` 控制按秒取帧；只读帧不写回视频。
- 每帧以 BGR 形式交给下游，并记录**时间戳（秒）**，用于 SQLite 与展示。
- **处理时长**：抽帧迭代器接受可选参数 `max_duration_sec`（单位：秒）。未指定时处理整段视频；指定时，当 `frame_time_sec >= max_duration_sec` 即停止抽帧，不再读取后续帧。

## 3. Triton 调用与人脸入库

- **复用** [triton_deploy/client/face_client.py](triton_deploy/client/face_client.py) 中的 `FaceTritonClient`（`get_faces(img_bgr)` 返回 bbox、score、kps、**embedding 512 维**）。
- 在 demo 中通过 `sys.path` 或包内相对引用加入 `triton_deploy/client`，初始化 `FaceTritonClient(url=...)`，URL 可由环境变量或 CLI 参数传入。
- 对每一帧：
  - 调用 `get_faces(frame_bgr)`；
  - 对每张人脸：将 embedding 写入 SQLite；将人脸 crop（按 bbox 从原帧裁剪，可适当 padding）保存为图片，路径写入 SQLite（见下 schema）。

## 4. SQLite 与“最简单”向量存储

- **单库单文件**即可，例如 `demo/output/movie_faces.db`（或按视频 hash 分子目录也可，按你偏好）。
- **表设计建议**（最少字段）：
  - **videos**：`id`, `path`（电影路径）, `created_at`（可选）, `fps_used`（抽帧 FPS, 可选）。
  - **faces**：`id`, `video_id`, `frame_time_sec`, `embedding`（BLOB，512 × float32）, `crop_path`（相对或绝对路径，用于 HTML 展示）, `cluster_id`（聚类后写入，int，同一人同一值）。
- **向量检索**：不做复杂索引，采用“最简单实现”：
  - 聚类时：一次性 `SELECT embedding, id FROM faces WHERE video_id = ?` 加载该电影全部 embedding 到内存，用 numpy 做相似度计算与聚类（见下）；
  - 若后续要做“检索”（例如按 embedding 找最近邻），可在同库内用 numpy 暴力余弦相似度即可，不引入向量库。

## 5. 聚类（同电影内“同一人”）

- **目标**：把同一电影内的 faces 按“是否同一人”聚成若干类，每类一个 `cluster_id`。
- **最简单实现**：贪心聚类（numpy 即可）：
  - 顺序遍历 faces，当前 face 与已有每个 cluster 的**代表向量**（如该类内所有 embedding 的均值并 L2 归一化）算余弦相似度；
  - 若与某类相似度 > 阈值（如 0.5，可配置），则归入该类并更新该类代表向量；
  - 否则新建一类。
- 聚类完成后对 `faces` 表批量 `UPDATE cluster_id`。
- **可选**：若希望更稳（遮挡、侧脸多），可增加依赖 `scikit-learn`，用 DBSCAN（cosine 距离）或 AgglomerativeClustering 在 embedding 上聚类，再写回 `cluster_id`。

## 6. 静态 HTML 页面

- **生成时机**：聚类完成后，由同一脚本生成。
- **内容**：
  - 标题：电影路径或文件名；
  - 按 **cluster_id 聚合**，每个“人物”一行/一块；
  - **排序**：按该人物出现的**图片数量（faces 条数）倒序**；
  - 每人展示：数量 + 若干张人脸缩略图（直接引用已保存的 crop 路径；若 HTML 与 crop 同目录或相对路径一致，用相对路径即可）。
- **实现**：用 Python 拼 HTML 字符串或简单模板（如 `str.format`/小段 Jinja2），写入 `demo/output/<video_id>/index.html`（与 `faces/` 下 crop 同层级或约定好相对路径），不依赖后端服务。
- **打开方式**：脚本结尾打印提示，例如 “已生成 index.html，请用浏览器打开：file:///path/to/demo/output/xxx/index.html”。

## 7. CLI 入口与参数

- **入口**：`demo/__main__.py` 或 `demo/cli.py`，通过 `python -m demo /path/to/movie.mp4` 调用。
- **参数建议**：
  - 位置参数：电影路径（必须）；
  - `--fps`：抽帧间隔（每秒取几帧，默认 1）；
  - `--duration`：处理时长（单位：秒）。**不指定则处理整个视频**；指定则只处理视频前 N 秒，到点即结束抽帧。
  - `--triton-url`：Triton 地址（默认 `localhost:8000` 或从环境变量读）；
  - `--output-dir`：输出根目录（默认如 `demo/output`）；
  - `--similarity-threshold`：聚类相似度阈值（默认 0.5，可选）。
- 执行顺序：抽帧（受 `--duration` 约束）→ 逐帧 Triton → 写 SQLite + 写 crop 文件 → 聚类并更新 `cluster_id` → 生成 HTML → 打印打开说明。

## 8. 文件与模块划分建议


| 路径                      | 职责                                                                                           |
| ----------------------- | -------------------------------------------------------------------------------------------- |
| `demo/__init__.py`      | 包占位                                                                                          |
| `demo/cli.py`           | 解析参数、串联：抽帧 → Triton → 存库 → 聚类 → 生成 HTML                                                      |
| `demo/frame_extract.py` | 抽帧：`VideoCapture` + 按 FPS 取帧，可选 `max_duration_sec` 限制处理时长；yield (frame_bgr, time_sec)        |
| `demo/db.py`            | SQLite：建表、插入 video/face、更新 cluster_id、按 video_id 查 faces（含 embedding）                        |
| `demo/cluster.py`       | 贪心聚类：输入 list of (id, embedding)，输出 list of (face_id, cluster_id)；或封装“从 DB 读 + 写回 cluster_id” |
| `demo/html.py`          | 根据 DB 中某 video 的 faces + cluster 信息，生成 index.html 字符串并写入文件                                   |
| `demo/output/`          | 默认输出目录：库文件、按视频分的子目录（crop + index.html）                                                       |


（若你希望少文件，可把 `frame_extract`、`cluster`、`html` 合并进 `cli.py`，但拆开更易测。）

## 9. 与现有代码的关系

- **Triton**：不修改 [triton_deploy/client/face_client.py](triton_deploy/client/face_client.py)，仅在 demo 中通过 path 或包引用使用 `FaceTritonClient`，与 [main_triton.py](main_triton.py) 一致。
- **项目 data/**：不依赖；仅使用用户传入的电影路径与 demo 自己的 output 目录。

## 10. 验收要点

- 给定一条电影路径，一条命令跑通：抽帧 → Triton → SQLite + crop → 聚类 → 生成 HTML。
- 生成的 HTML 用浏览器打开后，能看到该电影中“人物”列表，按出现图片数量从多到少排列，每人有多张人脸缩略图。

