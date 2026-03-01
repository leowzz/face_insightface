"""SQLite persistence helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from .schemas import ClusterAssignment, ClusterView, FaceEmbeddingRow, FaceInsert, VideoRow


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl_type: str) -> None:
    cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
    names = {c["name"] for c in cols}
    if column not in names:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            fps_used REAL NOT NULL,
            duration_limit_sec REAL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL,
            frame_index INTEGER NOT NULL,
            frame_time_sec REAL NOT NULL,
            score REAL NOT NULL,
            bbox_x1 REAL NOT NULL,
            bbox_y1 REAL NOT NULL,
            bbox_x2 REAL NOT NULL,
            bbox_y2 REAL NOT NULL,
            blur_var REAL,
            bbox_w REAL,
            bbox_h REAL,
            has_kps INTEGER,
            pose_yaw REAL,
            pose_pitch REAL,
            pose_roll REAL,
            age REAL,
            gender INTEGER,
            crop_path TEXT NOT NULL,
            embedding BLOB NOT NULL,
            cluster_id INTEGER,
            FOREIGN KEY(video_id) REFERENCES videos(id)
        );

        CREATE INDEX IF NOT EXISTS idx_faces_video ON faces(video_id);
        CREATE INDEX IF NOT EXISTS idx_faces_video_cluster ON faces(video_id, cluster_id);
        """
    )

    # migration for existing DBs
    _ensure_column(conn, "faces", "blur_var", "REAL")
    _ensure_column(conn, "faces", "bbox_w", "REAL")
    _ensure_column(conn, "faces", "bbox_h", "REAL")
    _ensure_column(conn, "faces", "has_kps", "INTEGER")
    _ensure_column(conn, "faces", "pose_yaw", "REAL")
    _ensure_column(conn, "faces", "pose_pitch", "REAL")
    _ensure_column(conn, "faces", "pose_roll", "REAL")
    _ensure_column(conn, "faces", "age", "REAL")
    _ensure_column(conn, "faces", "gender", "INTEGER")
    conn.commit()


def insert_video(conn: sqlite3.Connection, path: str, fps_used: float, duration_limit_sec: float | None) -> VideoRow:
    cursor = conn.execute(
        "INSERT INTO videos(path, fps_used, duration_limit_sec) VALUES(?, ?, ?)",
        (path, fps_used, duration_limit_sec),
    )
    conn.commit()
    return VideoRow(
        id=int(cursor.lastrowid),
        path=path,
        fps_used=fps_used,
        duration_limit_sec=duration_limit_sec,
    )


def insert_face(conn: sqlite3.Connection, face: FaceInsert) -> int:
    embedding_bytes = face.embedding.astype(np.float32).tobytes()
    cursor = conn.execute(
        """
        INSERT INTO faces(
            video_id, frame_index, frame_time_sec, score,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            blur_var, bbox_w, bbox_h, has_kps,
            pose_yaw, pose_pitch, pose_roll, age, gender,
            crop_path, embedding
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            face.video_id,
            face.frame_index,
            face.frame_time_sec,
            face.score,
            face.bbox_x1,
            face.bbox_y1,
            face.bbox_x2,
            face.bbox_y2,
            face.blur_var,
            face.bbox_w,
            face.bbox_h,
            face.has_kps,
            face.pose_yaw,
            face.pose_pitch,
            face.pose_roll,
            face.age,
            face.gender,
            face.crop_path,
            embedding_bytes,
        ),
    )
    return int(cursor.lastrowid)


def commit(conn: sqlite3.Connection) -> None:
    conn.commit()


def load_face_embeddings(conn: sqlite3.Connection, video_id: int) -> list[FaceEmbeddingRow]:
    rows = conn.execute(
        "SELECT id, embedding FROM faces WHERE video_id = ? ORDER BY id ASC",
        (video_id,),
    ).fetchall()
    result: list[FaceEmbeddingRow] = []
    for row in rows:
        embedding = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        result.append(FaceEmbeddingRow(face_id=int(row["id"]), embedding=embedding))
    return result


def save_cluster_assignments(conn: sqlite3.Connection, assignments: list[ClusterAssignment]) -> None:
    conn.executemany(
        "UPDATE faces SET cluster_id = ? WHERE id = ?",
        [(a.cluster_id, a.face_id) for a in assignments],
    )
    conn.commit()


def load_cluster_views(conn: sqlite3.Connection, video_id: int, preview_limit: int = 10) -> list[ClusterView]:
    cluster_rows = conn.execute(
        """
        SELECT cluster_id, COUNT(*) AS face_count
        FROM faces
        WHERE video_id = ? AND cluster_id IS NOT NULL
        GROUP BY cluster_id
        ORDER BY face_count DESC, cluster_id ASC
        """,
        (video_id,),
    ).fetchall()

    views: list[ClusterView] = []
    for row in cluster_rows:
        cluster_id = int(row["cluster_id"])

        preview_rows = conn.execute(
            """
            SELECT crop_path, score, COALESCE(blur_var, 0) AS blur_var,
                   pose_yaw, pose_roll
            FROM faces
            WHERE video_id = ? AND cluster_id = ?
            ORDER BY COALESCE(blur_var, 0) DESC, score DESC, id ASC
            LIMIT ?
            """,
            (video_id, cluster_id, max(1, preview_limit)),
        ).fetchall()

        info_rows = conn.execute(
            """
            SELECT score, COALESCE(blur_var, 0) AS blur_var, COALESCE(bbox_w, 0) AS bbox_w,
                   COALESCE(bbox_h, 0) AS bbox_h, age, gender
            FROM faces
            WHERE video_id = ? AND cluster_id = ?
            ORDER BY COALESCE(blur_var, 0) DESC, score DESC, id ASC
            LIMIT 3
            """,
            (video_id, cluster_id),
        ).fetchall()

        if not info_rows:
            continue

        n = len(info_rows)
        avg_score = sum(float(r["score"] or 0.0) for r in info_rows) / n
        avg_blur_var = sum(float(r["blur_var"] or 0.0) for r in info_rows) / n
        avg_bbox_w = sum(float(r["bbox_w"] or 0.0) for r in info_rows) / n
        avg_bbox_h = sum(float(r["bbox_h"] or 0.0) for r in info_rows) / n

        ages = [float(r["age"]) for r in info_rows if r["age"] is not None]
        avg_age = (sum(ages) / len(ages)) if ages else None

        male_count = sum(1 for r in info_rows if r["gender"] == 1)
        female_count = sum(1 for r in info_rows if r["gender"] == 0)
        dominant_gender = None
        if male_count > female_count:
            dominant_gender = "male"
        elif female_count > male_count:
            dominant_gender = "female"

        views.append(
            ClusterView(
                cluster_id=cluster_id,
                face_count=int(row["face_count"]),
                preview_paths=[str(item["crop_path"]) for item in preview_rows],
                preview_labels=[
                    (
                        f"det_conf={float(item['score']):.3f} | "
                        f"blur={float(item['blur_var']):.1f} | "
                        f"pose_yaw={float(item['pose_yaw']):.3f} | "
                        f"pose_roll={float(item['pose_roll']):.3f}"
                        if item['pose_yaw'] is not None and item['pose_roll'] is not None
                        else f"det_conf={float(item['score']):.3f} | blur={float(item['blur_var']):.1f} | pose=N/A"
                    )
                    for item in preview_rows
                ],
                avg_score=avg_score,
                avg_blur_var=avg_blur_var,
                avg_bbox_w=avg_bbox_w,
                avg_bbox_h=avg_bbox_h,
                avg_age=avg_age,
                dominant_gender=dominant_gender,
            )
        )
    return views


def count_faces(conn: sqlite3.Connection, video_id: int) -> int:
    row = conn.execute("SELECT COUNT(*) AS total FROM faces WHERE video_id = ?", (video_id,)).fetchone()
    return int(row["total"])
