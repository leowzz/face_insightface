.PHONY: run

# ===== 可调整参数（每项上方有说明） =====

# 输入视频路径。
# 调法：改成你的视频相对路径或绝对路径，例如 VIDEO=data/xxx.mp4。
VIDEO ?= data/jbx.mp4

# 抽帧频率（每秒取多少帧）。
# 调法：值越大采样越密、耗时越长；建议 0.3~2，常用 1。
FPS ?= 0.5

# 处理时长（秒），从 START_TIME 开始向后处理。
# 调法：值越大覆盖片段越长、耗时越长；如 120/240/600。
DURATION ?= 240

# 起始时间（h:m:s）。
# 调法：例如 0:35:0 表示从第 35 分钟开始。
START_TIME ?= 0:35:0

# Triton 推理服务地址。
# 调法：按你的服务 IP:PORT 修改；本地可用 localhost:8000。
TRITON_URL ?= 192.168.177.20:8100

# 人脸检测置信度阈值（0~1）。
# 调法：越低检出越多（误检也可能增加），越高更严格；常用 0.55~0.75。
DET_CONFIDENCE ?= 0.65

# 最小人脸尺寸（像素，取人脸框的最短边）。
# 调法：越大越能过滤远景小脸；常用 60~120。
MIN_FACE_SIZE ?= 80

# 清晰度阈值（Laplacian 方差）。
# 调法：越高越严格过滤模糊脸；常用 80~180。
BLUR_THRESHOLD ?= 120

# 聚类相似度阈值（0~1）。
# 调法：越高越“保守”（更不容易合并不同人），常用 0.6~0.75。
SIMILARITY_THRESHOLD ?= 0.65

# 姿态 yaw 偏差阈值（鼻子到双眼距离不对称度，0~1）。
# 调法：越大越宽松（允许更多侧脸），越小越偏向正脸；常用 0.15~0.35。
MAX_POSE_YAW_DEV ?= 0.30

# 姿态 roll 角阈值（双眼连线倾斜角，度）。
# 调法：越大越宽松（允许更大歪头），越小越严格；常用 10~30。
MAX_POSE_ROLL_DEG ?= 25

run:
	uv run python -m demo $(VIDEO) \
	--fps $(FPS) \
	--duration $(DURATION) \
	--triton-url $(TRITON_URL) \
	--start-time $(START_TIME) \
	--det-confidence-threshold $(DET_CONFIDENCE) \
	--min-face-size $(MIN_FACE_SIZE) \
	--blur-threshold $(BLUR_THRESHOLD) \
	--similarity-threshold $(SIMILARITY_THRESHOLD) \
	--max-pose-yaw-dev $(MAX_POSE_YAW_DEV) \
	--max-pose-roll-deg $(MAX_POSE_ROLL_DEG)
