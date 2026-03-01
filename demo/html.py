"""Static HTML rendering for clustered faces."""

from __future__ import annotations

from html import escape
from pathlib import Path

from .schemas import ClusterView


def _fmt(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def _kv_table(items: list[tuple[str, str]]) -> str:
    rows = "".join(f"<tr><th>{escape(k)}</th><td>{escape(v)}</td></tr>" for k, v in items)
    return f'<table class="kv">{rows}</table>'


def render_html(
    video_path: str,
    clusters: list[ClusterView],
    total_faces: int,
    run_meta: dict[str, str] | None = None,
) -> str:
    run_meta = run_meta or {}

    summary_items = [
        ("最后运行时间", run_meta.get("run_time", "N/A")),
        ("总人脸数(入库)", str(total_faces)),
        ("聚类人数", str(len(clusters))),
    ]

    params_items = [
        ("fps", run_meta.get("sample_fps", "N/A")),
        ("start_time_sec", run_meta.get("start_time_sec", "N/A")),
        ("duration_sec", run_meta.get("max_duration_sec", "N/A")),
        ("triton_url", run_meta.get("triton_url", "N/A")),
        ("det_conf_threshold", run_meta.get("det_conf_threshold", "N/A")),
        ("min_face_size", run_meta.get("min_face_size", "N/A")),
        ("blur_var_threshold", run_meta.get("blur_var_threshold", "N/A")),
        ("similarity_threshold", run_meta.get("similarity_threshold", "N/A")),
        ("max_pose_yaw_dev", run_meta.get("max_pose_yaw_dev", "N/A")),
        ("max_pose_roll_deg", run_meta.get("max_pose_roll_deg", "N/A")),
    ]

    filter_items = [
        ("raw_detect_count", run_meta.get("raw_detect_count", "N/A")),
        ("accepted_count", run_meta.get("accepted_count", "N/A")),
        ("filtered_total", run_meta.get("filtered_count", "N/A")),
        ("filtered_by_score", run_meta.get("filtered_by_score", "N/A")),
        ("filtered_by_size", run_meta.get("filtered_by_size", "N/A")),
        ("filtered_by_empty_crop", run_meta.get("filtered_by_empty_crop", "N/A")),
        ("filtered_by_blur", run_meta.get("filtered_by_blur", "N/A")),
        ("filtered_by_pose", run_meta.get("filtered_by_pose", "N/A")),
    ]

    cards: list[str] = []
    for idx, cluster in enumerate(clusters, start=1):
        imgs_parts: list[str] = []
        for i, path in enumerate(cluster.preview_paths):
            if i < 3:
                label = cluster.preview_labels[i] if i < len(cluster.preview_labels) else "N/A"
                tag_html = label
                imgs_parts.append(
                    '<div class="img-wrap">'
                    f'<img src="{escape(path)}" alt="cluster-{cluster.cluster_id}" loading="lazy" />'
                    f'<div class="img-tag">TOP{i+1}<br/>{tag_html}</div>'
                    '</div>'
                )
            else:
                imgs_parts.append(
                    '<div class="img-wrap">'
                    f'<img src="{escape(path)}" alt="cluster-{cluster.cluster_id}" loading="lazy" />'
                    '</div>'
                )
        imgs = "".join(imgs_parts)
        cards.append(
            '<section class="card">'
            f"<h2>人物 {idx}</h2>"
            f"<p>cluster_id={cluster.cluster_id} | 图片数={cluster.face_count} | 页面展示最清晰 {len(cluster.preview_paths)} 张（最多10张，前3张标注参数对应值）</p>"
            '<ul class="facts">'
            f"<li>基础信息(取最清晰3张) 平均检测置信度: {_fmt(cluster.avg_score, 3)}</li>"
            f"<li>平均清晰度(拉普拉斯方差): {_fmt(cluster.avg_blur_var, 1)}</li>"
            f"<li>平均人脸框尺寸: {_fmt(cluster.avg_bbox_w, 0)} × {_fmt(cluster.avg_bbox_h, 0)}</li>"
            f"<li>平均姿态 yaw/roll(若可用): {_fmt(cluster.avg_pose_yaw, 3)} / {_fmt(cluster.avg_pose_roll, 3)}</li>"
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
    .meta {{ color: #666; margin-bottom: 16px; }}
    .panel {{ background: #fff; border-radius: 12px; padding: 14px 16px; margin-bottom: 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    .card {{ background: #fff; border-radius: 12px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    .facts {{ margin: 8px 0 10px; color: #444; }}
    .facts li {{ margin: 2px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 8px; margin-top: 12px; }}
    .grid img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 8px; border: 1px solid #eee; }}
    .kv {{ width: 100%; border-collapse: collapse; }}
    .kv th, .kv td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #f0f0f0; font-size: 14px; }}
    .kv th {{ color: #555; width: 260px; }}
  </style>
</head>
<body>
  <h1>电影人脸聚类结果</h1>
  <p class="meta">视频: {escape(video_path)}</p>

  <section class="panel"><h3>运行概览</h3>{_kv_table(summary_items)}</section>
  <section class="panel"><h3>本次运行参数</h3>{_kv_table(params_items)}</section>
  <section class="panel"><h3>人脸统计与过滤明细</h3>{_kv_table(filter_items)}</section>

  {body_html}
</body>
</html>
"""


def write_html(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
