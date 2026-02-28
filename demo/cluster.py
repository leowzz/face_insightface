"""Simple greedy face clustering."""

from __future__ import annotations

import numpy as np

from .schemas import ClusterAssignment, FaceEmbeddingRow


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec
    return vec / norm


def greedy_cluster(
    rows: list[FaceEmbeddingRow],
    similarity_threshold: float,
) -> list[ClusterAssignment]:
    """Cluster embeddings with greedy centroid matching.

    :param rows: 人脸向量列表。
    :param similarity_threshold: 余弦相似度阈值。
    :return: 每个 face 的 cluster 归属。
    """

    if not rows:
        return []

    centroids: list[np.ndarray] = []
    counts: list[int] = []
    assignments: list[ClusterAssignment] = []

    for row in rows:
        emb = _l2_normalize(row.embedding.astype(np.float32))
        if not centroids:
            centroids.append(emb.copy())
            counts.append(1)
            assignments.append(ClusterAssignment(face_id=row.face_id, cluster_id=1))
            continue

        sims = [float(np.dot(emb, center)) for center in centroids]
        best_idx = int(np.argmax(np.asarray(sims, dtype=np.float32)))
        best_sim = sims[best_idx]

        if best_sim >= similarity_threshold:
            cluster_idx = best_idx
            updated = (centroids[cluster_idx] * counts[cluster_idx] + emb) / (counts[cluster_idx] + 1)
            centroids[cluster_idx] = _l2_normalize(updated)
            counts[cluster_idx] += 1
            assignments.append(ClusterAssignment(face_id=row.face_id, cluster_id=cluster_idx + 1))
        else:
            centroids.append(emb.copy())
            counts.append(1)
            assignments.append(ClusterAssignment(face_id=row.face_id, cluster_id=len(centroids)))

    return assignments
