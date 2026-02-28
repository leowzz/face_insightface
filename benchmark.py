import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis

REPEAT = 10
IMG_PATH = "data/leo01.png"


def mean_std(times):
    arr = np.array(times)
    return arr.mean() * 1000, arr.std() * 1000  # ms


def main():
    print("初始化模型（首次加载，不计入耗时）...")
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    img = cv2.imread(IMG_PATH)
    assert img is not None, f"无法读取 {IMG_PATH}"

    # 预热
    for _ in range(3):
        app.get(img)

    print(f"\n测试图片: {IMG_PATH}  重复次数: {REPEAT}\n")
    print("=" * 50)

    # 1. 完整流程耗时（检测 + 特征提取）
    times_full = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        faces = app.get(img)
        times_full.append(time.perf_counter() - t0)

    m, s = mean_std(times_full)
    print(f"完整推理（检测+特征提取）: {m:.1f} ± {s:.1f} ms")

    # 2. 仅检测耗时（关闭 recognition 模型）
    app_det = FaceAnalysis(allowed_modules=["detection"], providers=["CPUExecutionProvider"])
    app_det.prepare(ctx_id=0, det_size=(640, 640))
    for _ in range(3):
        app_det.get(img)

    times_det = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        app_det.get(img)
        times_det.append(time.perf_counter() - t0)

    m2, s2 = mean_std(times_det)
    print(f"仅检测（detection only）  : {m2:.1f} ± {s2:.1f} ms")

    # 3. 仅特征提取耗时（在已检测人脸上单独跑 recognition）
    faces = app.get(img)
    face = faces[0]
    recognizer = app.models["recognition"]
    # 预热
    for _ in range(3):
        recognizer.get(img, face)

    times_rec = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        recognizer.get(img, face)
        times_rec.append(time.perf_counter() - t0)

    m3, s3 = mean_std(times_rec)
    print(f"仅特征提取（recognition） : {m3:.1f} ± {s3:.1f} ms")

    print("=" * 50)
    print(f"检测占比: {m2/m*100:.1f}%   特征提取占比: {m3/m*100:.1f}%")

    # 4. 不同图像分辨率下的检测耗时
    print("\n--- 不同 det_size 对比 ---")
    for det_size in [(320, 320), (480, 480), (640, 640)]:
        a = FaceAnalysis(allowed_modules=["detection"], providers=["CPUExecutionProvider"])
        a.prepare(ctx_id=0, det_size=det_size)
        for _ in range(3):
            a.get(img)
        ts = []
        for _ in range(REPEAT):
            t0 = time.perf_counter()
            a.get(img)
            ts.append(time.perf_counter() - t0)
        m, s = mean_std(ts)
        print(f"  det_size={det_size}: {m:.1f} ± {s:.1f} ms")


if __name__ == "__main__":
    main()
