.PHONY: run

run:
	uv run python -m demo data/jbx.mp4 \
	--fps 0.3 --duration 600 --triton-url 192.168.177.20:8100 --start-time 0:5:0 --blur-threshold 120 --max-pose-yaw-dev 0.18 --max-pose-roll-deg 15 
