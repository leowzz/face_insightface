"""Static HTML rendering for clustered faces."""

from __future__ import annotations

from html import escape
from pathlib import Path

from .schemas import ClusterView


def _fmt(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def render_html(
    video_path: str,
    clusters: list[ClusterView],
    total_faces: int,
) -> str:
    cards: list[str] = []
    for idx, cluster in enumerate(clusters, start=1):
        imgs = "".join(
            f'<img src="{escape(path)}" alt="cluster-{cluster.cluster_id}" loading="lazy" />'
            for path in cluster.preview_paths
        )
        cards.append(
            '<section class="card">'
            f"<h2>人物 {idx}</h2>"
            f"<p>cluster_id={cluster.cluster_id} | 图片数={cluster.face_count} | 页面展示最清晰 {len(cluster.preview_paths)} 张（最多10张）</p>"
            '<ul class="facts">'
            f"<li>基础信息(取最清晰3张) 平均检测置信度: {_fmt(cluster.avg_score, 3)}</li>"
            f"<li>平均清晰度(拉普拉斯方差): {_fmt(cluster.avg_blur_var, 1)}</li>"
            f"<li>平均人脸框尺寸: {_fmt(cluster.avg_bbox_w, 0)} × {_fmt(cluster.avg_bbox_h, 0)}</li>"
            f"<li>年龄(若可用): {_fmt(cluster.avg_age, 1)}</li>"
            f"<li>性别(若可用): {escape(cluster.dominant_gender or 'N/A')}</li>"
            '</ul>'
            f'<div class="grid">{imgs}</div>'
            "</section>"
        )

    body_html = "".join(cards) if cards else "<p>未检测到可聚类的人脸。</p>"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>电影人脸聚类结果</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f7f7f8; color: #222; }}
    h1 {{ margin: 0 0 8px; }}
    .meta {{ color: #666; margin-bottom: 24px; }}
    .card {{ background: #fff; border-radius: 12px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    .facts {{ margin: 8px 0 10px; color: #444; }}
    .facts li {{ margin: 2px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 8px; margin-top: 12px; }}
    .grid img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 8px; border: 1px solid #eee; }}
  </style>
</head>
<body>
  <h1>电影人脸聚类结果</h1>
  <p class="meta">视频: {escape(video_path)}<br/>总人脸数: {total_faces} | 聚类人数: {len(clusters)}</p>
  {body_html}
</body>
</html>
"""


def write_html(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
