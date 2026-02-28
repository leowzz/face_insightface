#!/usr/bin/env bash
# prepare_models.sh - 复制模型文件并打印真实的 tensor 名称和 shape
set -euo pipefail

INSIGHTFACE_DIR="${HOME}/.insightface/models/buffalo_l"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)/model_repository"

echo "=== 复制模型文件 ==="
cp "${INSIGHTFACE_DIR}/det_10g.onnx"   "${REPO_DIR}/det_10g/1/model.onnx"
cp "${INSIGHTFACE_DIR}/w600k_r50.onnx" "${REPO_DIR}/w600k_r50/1/model.onnx"
echo "已复制 det_10g.onnx → model_repository/det_10g/1/model.onnx"
echo "已复制 w600k_r50.onnx → model_repository/w600k_r50/1/model.onnx"

echo ""
echo "=== 打印真实 Tensor 名称（请对照核实 config.pbtxt）==="
uv run --with onnxruntime python - <<'PYEOF'
import onnxruntime as ort
import os

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_repository")

for model_name, path in [
    ("det_10g",   os.path.join(base, "det_10g/1/model.onnx")),
    ("w600k_r50", os.path.join(base, "w600k_r50/1/model.onnx")),
]:
    print(f"\n--- {model_name} ---")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print("Inputs:")
    for inp in sess.get_inputs():
        print(f"  name={inp.name!r}  shape={inp.shape}  dtype={inp.type}")
    print("Outputs:")
    for out in sess.get_outputs():
        print(f"  name={out.name!r}  shape={out.shape}  dtype={out.type}")
PYEOF

echo ""
echo "=== 完成 ==="
echo "请用上述输出核对 config.pbtxt 中的 input/output name 字段"
