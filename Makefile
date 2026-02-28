.PHONY: run

run:
	uv run python -m demo data/jbx.mp4 \
	--fps 0.3 --duration 300 --triton-url 192.168.177.20:8100 --start-time 0:5:0 
