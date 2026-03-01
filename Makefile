.PHONY: run

run:
	uv run python -m demo data/jbx.mp4 \
	--fps 1 --duration 120 --triton-url 192.168.177.20:8100 --start-time 0:35:0 --blur-threshold 120 --max-pose-yaw-dev 0.18 --max-pose-roll-deg 15 
