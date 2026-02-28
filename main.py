import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def draw_faces(img, faces, labels=None):
    img_out = img.copy()
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = labels[i] if labels and i < len(labels) else f"face{i}"
        cv2.putText(img_out, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if face.kps is not None:
            for kp in face.kps.astype(int):
                cv2.circle(img_out, tuple(kp), 2, (0, 0, 255), -1)
    return img_out


def main():
    data_dir = "data"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 初始化模型
    print("正在初始化 InsightFace 模型...")
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("模型初始化完成\n")

    # 单人照片文件列表
    single_files = {
        "leo01": "leo01.png",
        "leo02": "leo02.png",
        "leo03": "leo03.png",
        "qi01":  "qi01.png",
        "qi02":  "qi02.png",
    }

    # 同框照片文件列表
    group_files = {
        "leo_qi_01": "leo_qi_01.png",
        "leo_qi_02": "leo_qi_02.png",
    }

    # ── Step 1: 检测单人照 + 提取 embedding ──────────────────────────────
    print("=" * 50)
    print("【单人照片人脸检测 & 特征提取】")
    print("=" * 50)

    embeddings = {}
    for name, fname in single_files.items():
        path = os.path.join(data_dir, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"  [警告] 无法读取 {path}")
            continue

        faces = app.get(img)
        if not faces:
            print(f"  {name}: 未检测到人脸")
            continue

        face = faces[0]
        embeddings[name] = face.embedding
        print(f"  {name}: 检测到 {len(faces)} 张人脸，embedding 维度={face.embedding.shape[0]}")

        # 可视化并保存
        img_out = draw_faces(img, [face], labels=[name])
        cv2.imwrite(os.path.join(output_dir, fname), img_out)

    # ── Step 2: 相似度矩阵 ───────────────────────────────────────────────
    print()
    print("=" * 50)
    print("【人脸相似度矩阵（余弦相似度）】")
    print("=" * 50)

    names = list(embeddings.keys())
    n = len(names)
    col_w = 10
    header = " " * col_w + "".join(f"{n:>{col_w}}" for n in names)
    print(header)

    for i, ni in enumerate(names):
        row = f"{ni:<{col_w}}"
        for j, nj in enumerate(names):
            sim = cosine_similarity(embeddings[ni], embeddings[nj])
            row += f"{sim:>{col_w}.4f}"
        print(row)

    # 打印关键对比结论
    print()
    leo_keys = [k for k in names if k.startswith("leo")]
    qi_keys  = [k for k in names if k.startswith("qi")]

    leo_sims = [cosine_similarity(embeddings[a], embeddings[b])
                for a in leo_keys for b in leo_keys if a != b]
    cross_sims = [cosine_similarity(embeddings[a], embeddings[b])
                  for a in leo_keys for b in qi_keys]

    if leo_sims:
        print(f"  Leo 同人平均相似度:   {np.mean(leo_sims):.4f}")
    if cross_sims:
        print(f"  Leo vs Qi 平均相似度: {np.mean(cross_sims):.4f}")

    # ── Step 3: 同框图片验证 ─────────────────────────────────────────────
    print()
    print("=" * 50)
    print("【同框图片人脸识别验证】")
    print("=" * 50)

    # 取 leo/qi 各一个参考 embedding
    ref = {}
    if "leo01" in embeddings:
        ref["leo"] = embeddings["leo01"]
    if "qi01" in embeddings:
        ref["qi"]  = embeddings["qi01"]

    for gname, fname in group_files.items():
        path = os.path.join(data_dir, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"  [警告] 无法读取 {path}")
            continue

        faces = app.get(img)
        print(f"\n  {gname}: 检测到 {len(faces)} 张人脸")

        labels = []
        for idx, face in enumerate(faces):
            best_name, best_sim = "unknown", -1.0
            for rname, remb in ref.items():
                sim = cosine_similarity(face.embedding, remb)
                if sim > best_sim:
                    best_sim = sim
                    best_name = rname
            label = f"{best_name}({best_sim:.2f})"
            labels.append(label)
            print(f"    人脸#{idx}: 最相似={best_name}  相似度={best_sim:.4f}")

        img_out = draw_faces(img, faces, labels=labels)
        cv2.imwrite(os.path.join(output_dir, fname), img_out)

    print()
    print("=" * 50)
    print(f"可视化结果已保存到 {output_dir}/ 目录")
    print("=" * 50)


if __name__ == "__main__":
    main()
