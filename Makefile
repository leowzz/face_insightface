.PHONY: run

# ===== 可调整参数 =====
VIDEO ?= data/jbx.mp4
FPS ?= 1
DURATION ?= 120
START_TIME ?= 0:35:0
TRITON_URL ?= 192.168.177.20:8100

# 质量与聚类参数
DET_CONFIDENCE ?= 0.585
MIN_FACE_SIZE ?= 80
BLUR_THRESHOLD ?= 120
SIMILARITY_THRESHOLD ?= 0.65

# 姿态过滤（本次降低阈值，更严格）
MAX_POSE_YAW_DEV ?= 0.15
MAX_POSE_ROLL_DEG ?= 12

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
