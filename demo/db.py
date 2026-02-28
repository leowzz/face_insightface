"""SQLite persistence helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from .schemas import ClusterAssignment, ClusterView, FaceEmbeddingRow, FaceInsert, VideoRow


def connect_db(db_path: Path) -> sqlite3.Connection:
    """Create sqlite connection.

    :param db_path: SQLite 文件路径。
    :return: sqlite 连接。
    """

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Initialize required tables."""

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
            crop_path TEXT NOT NULL,
            embedding BLOB NOT NULL,
            cluster_id INTEGER,
            FOREIGN KEY(video_id) REFERENCES videos(id)
        );

        CREATE INDEX IF NOT EXISTS idx_faces_video ON faces(video_id);
        CREATE INDEX IF NOT EXISTS idx_faces_video_cluster ON faces(video_id, cluster_id);
        """
    )
    conn.commit()


def insert_video(conn: sqlite3.Connection, path: str, fps_used: float, duration_limit_sec: float | None) -> VideoRow:
    """Insert a video record and return schema object."""

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
    """Insert a face row and return row id."""

    embedding_bytes = face.embedding.astype(np.float32).tobytes()
    cursor = conn.execute(
        """
        INSERT INTO faces(
            video_id, frame_index, frame_time_sec, score,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            crop_path, embedding
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            face.crop_path,
            embedding_bytes,
        ),
    )
    return int(cursor.lastrowid)


def commit(conn: sqlite3.Connection) -> None:
    """Commit transaction."""

    conn.commit()


def load_face_embeddings(conn: sqlite3.Connection, video_id: int) -> list[FaceEmbeddingRow]:
    """Load all embeddings for a video."""

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
    """Persist clustering results."""

    conn.executemany(
        "UPDATE faces SET cluster_id = ? WHERE id = ?",
        [(a.cluster_id, a.face_id) for a in assignments],
    )
    conn.commit()


def load_cluster_views(conn: sqlite3.Connection, video_id: int, preview_limit: int = 12) -> list[ClusterView]:
    """Load cluster summary data for rendering."""

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
            SELECT crop_path
            FROM faces
            WHERE video_id = ? AND cluster_id = ?
            ORDER BY score DESC, id ASC
            LIMIT ?
            """,
            (video_id, cluster_id, preview_limit),
        ).fetchall()
        views.append(
            ClusterView(
                cluster_id=cluster_id,
                face_count=int(row["face_count"]),
                preview_paths=[str(item["crop_path"]) for item in preview_rows],
            )
        )
    return views


def count_faces(conn: sqlite3.Connection, video_id: int) -> int:
    """Count total faces for video."""

    row = conn.execute("SELECT COUNT(*) AS total FROM faces WHERE video_id = ?", (video_id,)).fetchone()
    return int(row["total"])
