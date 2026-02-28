# triton_deploy Makefile
# 使用方式: make run  （需先 docker compose up -d 并安装 client 依赖）
# 可选: make run IMG_PATH=/path/to/image.png TRITON_SERVER=host:port

ROOT_DIR := $(abspath ..)
DATA_DIR := $(ROOT_DIR)/data
# 测试用图像绝对路径；未填则使用 data 目录下第一张图
IMG_PATH ?=
# Triton Server 地址，格式 host:port
TRITON_SERVER ?= localhost:8000

# 未指定 IMG_PATH 时，取 data 目录下第一张图片（按文件名排序）
FIRST_IMG := $(shell ls $(DATA_DIR)/*.png $(DATA_DIR)/*.jpg $(DATA_DIR)/*.jpeg 2>/dev/null | head -1)
RUN_IMG := $(if $(IMG_PATH),$(IMG_PATH),$(FIRST_IMG))

.PHONY: run
run:
	@if [ -z "$(RUN_IMG)" ]; then \
		echo "未找到图片：请设置 IMG_PATH 或在项目 data 目录下放置至少一张图片（.png / .jpg / .jpeg）"; \
		exit 1; \
	fi
	@echo "使用图片: $(RUN_IMG)"
	@echo "Triton Server: $(TRITON_SERVER)"
	@cd client && TRITON_SERVER_URL="$(TRITON_SERVER)" python face_client.py "$(RUN_IMG)"
