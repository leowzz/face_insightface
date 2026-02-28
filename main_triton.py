import os
import sys
import cv2
import numpy as np


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def draw_faces(img, faces, labels=None):
    img_out = img.copy()
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face["bbox"].astype(int)
        cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = labels[i] if labels and i < len(labels) else f"face{i}"
        cv2.putText(
            img_out,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        if face["kps"] is not None:
            for kp in face["kps"].astype(int):
                cv2.circle(img_out, tuple(kp), 2, (0, 0, 255), -1)
    return img_out


def main():
    data_dir = "data"
    output_dir = "output_triton"
    os.makedirs(output_dir, exist_ok=True)

    # 引入 Triton 客户端实现
    client_dir = os.path.join(os.path.dirname(__file__), "triton_deploy", "client")
    if client_dir not in sys.path:
        sys.path.append(client_dir)
    from face_client import FaceTritonClient

    print("正在初始化 Triton 客户端...")
    app = FaceTritonClient(url="localhost:8000", det_size=(640, 640))
    print("Triton 客户端初始化完成\n")

    single_files = {
        "leo01": "leo01.png",
        "leo02": "leo02.png",
        "leo03": "leo03.png",
        "qi01": "qi01.png",
        "qi02": "qi02.png",
    }

    group_files = {
        "leo_qi_01": "leo_qi_01.png",
        "leo_qi_02": "leo_qi_02.png",
    }

    print("=" * 50)
    print("【单人照片人脸检测 & 特征提取（Triton）】")
    print("=" * 50)

    embeddings = {}
    for name, fname in single_files.items():
        path = os.path.join(data_dir, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"  [警告] 无法读取 {path}")
            continue

        faces = app.get_faces(img)
        if not faces:
            print(f"  {name}: 未检测到人脸")
            continue

        face = faces[0]
        embeddings[name] = face["embedding"]
        print(
            f"  {name}: 检测到 {len(faces)} 张人脸，embedding 维度={face['embedding'].shape[0]}"
        )

        img_out = draw_faces(img, [face], labels=[name])
        cv2.imwrite(os.path.join(output_dir, fname), img_out)

    print()
    print("=" * 50)
    print("【人脸相似度矩阵（余弦相似度）】")
    print("=" * 50)

    names = list(embeddings.keys())
    col_w = 10
    header = " " * col_w + "".join(f"{n:>{col_w}}" for n in names)
    print(header)

    for ni in names:
        row = f"{ni:<{col_w}}"
        for nj in names:
            sim = cosine_similarity(embeddings[ni], embeddings[nj])
            row += f"{sim:>{col_w}.4f}"
        print(row)

    print()
    leo_keys = [k for k in names if k.startswith("leo")]
    qi_keys = [k for k in names if k.startswith("qi")]

    leo_sims = [
        cosine_similarity(embeddings[a], embeddings[b])
        for a in leo_keys
        for b in leo_keys
        if a != b
    ]
    cross_sims = [
        cosine_similarity(embeddings[a], embeddings[b]) for a in leo_keys for b in qi_keys
    ]

    if leo_sims:
        print(f"  Leo 同人平均相似度:   {np.mean(leo_sims):.4f}")
    if cross_sims:
        print(f"  Leo vs Qi 平均相似度: {np.mean(cross_sims):.4f}")

    print()
    print("=" * 50)
    print("【同框图片人脸识别验证（Triton）】")
    print("=" * 50)

    ref = {}
    if "leo01" in embeddings:
        ref["leo"] = embeddings["leo01"]
    if "qi01" in embeddings:
        ref["qi"] = embeddings["qi01"]

    for gname, fname in group_files.items():
        path = os.path.join(data_dir, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"  [警告] 无法读取 {path}")
            continue

        faces = app.get_faces(img)
        print(f"\n  {gname}: 检测到 {len(faces)} 张人脸")

        labels = []
        for idx, face in enumerate(faces):
            best_name, best_sim = "unknown", -1.0
            for rname, remb in ref.items():
                sim = cosine_similarity(face["embedding"], remb)
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
