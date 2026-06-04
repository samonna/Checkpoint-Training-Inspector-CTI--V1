"""
report.py – Enhanced HTML report generation for Checkpoint Training Inspector V3.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _narrative(summary: Dict, health: Dict, anomalies_df: pd.DataFrame) -> str:
    parts: List[str] = []
    grc = summary.get("global_relative_change", float("nan"))
    comp = summary.get("comparable_layers", 0)
    issues = summary.get("issues", 0)

    if not isinstance(grc, float) or np.isnan(grc):
        parts.append("Global drift could not be computed.")
    elif grc < 0.001:
        parts.append(f"<b>Very low global drift ({grc:.5f})</b> — checkpoints are nearly identical. Consider whether training is progressing.")
    elif grc < 0.05:
        parts.append(f"<b>Moderate global drift ({grc:.5f})</b> — healthy parameter movement across {comp} comparable layers.")
    else:
        parts.append(f"<b>High global drift ({grc:.5f})</b> — significant shifts detected. Verify this is expected.")

    frozen = health.get("frozen_layers", 0)
    exploding = health.get("exploding_layers", 0)
    cos_collapse = health.get("cosine_collapse_layers", 0)
    top_layer = health.get("most_changed_layer", "N/A")

    if frozen > 0:
        parts.append(f"<b>{frozen} frozen layer(s)</b> — relative change &lt; 1e-6. Verify intentional freezing (e.g. backbone).")
    if exploding > 0:
        parts.append(f"<b>{exploding} exploding layer(s)</b> — relative change &gt; 5.0. Check gradient clipping and LR.")
    if cos_collapse > 0:
        parts.append(f"<b>{cos_collapse} cosine collapse layer(s)</b> — representations reversed direction.")
    if top_layer and top_layer != "N/A":
        parts.append(f"Most-changed layer: <code>{top_layer}</code>.")
    if issues > 0:
        parts.append(f"<b>{issues} architecture issue(s)</b> — missing or shape-mismatched layers.")

    if not anomalies_df.empty and "event" in anomalies_df.columns:
        spikes = anomalies_df[anomalies_df["event"].str.contains("spike|collapse", na=False)]
        plateaus = anomalies_df[anomalies_df["event"] == "plateau"]
        if not spikes.empty:
            parts.append(f"<b>Trajectory instability</b> at {len(spikes)} checkpoint(s).")
        if not plateaus.empty:
            parts.append("<b>Training plateau</b> detected.")

    return "<br>".join(f"• {p}" for p in parts) if parts else "No significant anomalies detected."


def _sparkline_svg(values: List[float], width: int = 400, height: int = 60) -> str:
    if len(values) < 2:
        return ""
    vmin, vmax = min(values), max(values)
    rng = vmax - vmin or 1.0
    xs = [int(i / (len(values) - 1) * (width - 10)) + 5 for i in range(len(values))]
    ys = [int((1 - (v - vmin) / rng) * (height - 10)) + 5 for v in values]
    points = " ".join(f"{x},{y}" for x, y in zip(xs, ys))
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<polyline points="{points}" fill="none" stroke="#4f8ef7" stroke-width="2"/>'
        f'</svg>'
    )


def _colour_row(rc: float) -> str:
    if rc > 5.0:
        return "background:#ffd6d6"
    if rc < 1e-6:
        return "background:#e8f4e8"
    if rc > 1.0:
        return "background:#fff3cd"
    return ""


def _df_to_coloured_html(df: pd.DataFrame, n: int = 40) -> str:
    if df.empty:
        return "<p><em>None.</em></p>"
    headers = df.columns.tolist()
    rows_html = "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
    for _, row in df.head(n).iterrows():
        rc = row.get("relative_change", 0.5)
        style = _colour_row(float(rc)) if isinstance(rc, (int, float)) and not np.isnan(float(rc)) else ""
        cells = "".join(f'<td style="{style}">{str(v)[:80]}</td>' for v in row.values)
        rows_html += f"<tr>{cells}</tr>"
    return f"<table>{rows_html}</table>"


def html_report(
    title: str,
    summary: Dict[str, object],
    layer_df: pd.DataFrame,
    group_df: pd.DataFrame,
    issues_df: pd.DataFrame,
    *,
    health: Optional[Dict] = None,
    anomalies_df: Optional[pd.DataFrame] = None,
    trajectory_values: Optional[List[float]] = None,
) -> bytes:
    health = health or {}
    anomalies_df = anomalies_df if anomalies_df is not None else pd.DataFrame()
    trajectory_values = trajectory_values or []

    narrative = _narrative(summary, health, anomalies_df)
    sparkline = _sparkline_svg(trajectory_values) if trajectory_values else ""
    top_layers_html = _df_to_coloured_html(layer_df)
    top_groups_html = _df_to_coloured_html(group_df)
    issues_html = issues_df.head(50).to_html(index=False, escape=True) if not issues_df.empty else "<p><em>None.</em></p>"
    anomalies_html = anomalies_df.to_html(index=False, escape=True) if not anomalies_df.empty else "<p><em>None.</em></p>"
    health_items = "".join(f"<li><b>{k.replace('_',' ').title()}</b>: {v}</li>" for k, v in health.items())
    summary_items = "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in summary.items())
    colour_legend = """<p style="font-size:11px">
<span style="background:#ffd6d6;padding:2px 6px">Exploding (RC&gt;5)</span>&nbsp;
<span style="background:#fff3cd;padding:2px 6px">High drift (RC&gt;1)</span>&nbsp;
<span style="background:#e8f4e8;padding:2px 6px">Frozen (RC&lt;1e-6)</span></p>"""

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>
body{{font-family:Arial,sans-serif;margin:40px;max-width:1200px;color:#222}}
h1{{color:#2c3e50;border-bottom:2px solid #4f8ef7;padding-bottom:8px}}
h2{{color:#34495e;margin-top:32px}}
table{{border-collapse:collapse;width:100%;font-size:11px;margin-top:8px}}
th{{background:#2c3e50;color:white;padding:6px 8px;text-align:left}}
td{{border:1px solid #ddd;padding:5px 8px}}
code{{background:#f0f0f0;padding:1px 4px;border-radius:3px;font-size:12px}}
.narrative{{background:#eef4ff;border-left:4px solid #4f8ef7;padding:12px 16px;border-radius:4px;line-height:1.8}}
details{{margin:8px 0}} summary{{cursor:pointer;font-weight:bold;color:#4f8ef7;padding:4px 0}}
ul{{line-height:1.8}}
</style></head><body>
<h1>{title}</h1>
<p style="color:#888">Generated: {datetime.now().isoformat(timespec='seconds')}</p>
<h2>Automated Narrative</h2>
<div class="narrative">{narrative}</div>
<h2>Summary Statistics</h2><ul>{summary_items}</ul>
<h2>Layer Health Diagnostics</h2><ul>{health_items}</ul>
{f'<h2>Global Drift Trajectory</h2>{sparkline}' if sparkline else ''}
<h2>Trajectory Anomalies</h2>{anomalies_html}
<h2>Top Layer Differences</h2>{colour_legend}
<details open><summary>Layer table</summary>{top_layers_html}</details>
<h2>Grouped Module Drift</h2>
<details open><summary>Group table</summary>{top_groups_html}</details>
<h2>Architecture Issues</h2>{issues_html}
</body></html>"""
    return html.encode("utf-8")
