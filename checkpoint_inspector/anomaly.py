"""
anomaly.py – Training anomaly detection for Checkpoint Training Inspector V3.

Detects:
- Sudden spike / collapse in drift between consecutive checkpoints
- Dead layers (near-zero weight norm, near-zero drift)
- Exploding layers (relative_change > threshold)
- Frozen layers (relative_change below noise floor)
- Cosine similarity collapse (representation reversal)
- Weight norm explosion / vanishing
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────────────────
# Per-checkpoint-pair anomalies
# ────────────────────────────────────────────────────────────────────────────

def detect_layer_anomalies(
    df: pd.DataFrame,
    *,
    frozen_threshold: float = 1e-6,
    exploding_threshold: float = 5.0,
    cosine_collapse_threshold: float = 0.5,
    dead_norm_threshold: float = 1e-7,
) -> pd.DataFrame:
    """Flag anomalous layers in a layer-comparison DataFrame.

    Parameters
    ----------
    df : layer DataFrame from compare_state_dicts
    frozen_threshold : relative_change below this → frozen
    exploding_threshold : relative_change above this → exploding
    cosine_collapse_threshold : cosine_similarity below this → collapse
    dead_norm_threshold : a_norm or b_norm below this → dead layer

    Returns a DataFrame of flagged layers with a 'flags' column.
    """
    if df.empty:
        return pd.DataFrame(columns=["layer", "flags", "relative_change", "cosine_similarity", "a_norm", "b_norm"])

    rows: List[Dict] = []
    for _, row in df.iterrows():
        flags = []
        rc = row.get("relative_change", float("nan"))
        cos = row.get("cosine_similarity", float("nan"))
        a_norm = row.get("a_norm", float("nan"))
        b_norm = row.get("b_norm", float("nan"))

        if not np.isnan(rc):
            if rc < frozen_threshold:
                flags.append("frozen")
            if rc > exploding_threshold:
                flags.append("exploding")
        if not np.isnan(cos) and cos < cosine_collapse_threshold:
            flags.append("cosine_collapse")
        if not np.isnan(a_norm) and a_norm < dead_norm_threshold:
            flags.append("dead_in_A")
        if not np.isnan(b_norm) and b_norm < dead_norm_threshold:
            flags.append("dead_in_B")

        if flags:
            rows.append({
                "layer": row["layer"],
                "flags": ", ".join(flags),
                "relative_change": rc,
                "cosine_similarity": cos,
                "a_norm": a_norm,
                "b_norm": b_norm,
            })

    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# Trajectory-level anomalies
# ────────────────────────────────────────────────────────────────────────────

def detect_trajectory_anomalies(
    traj_df: pd.DataFrame,
    *,
    spike_z: float = 2.5,
    plateau_window: int = 3,
    plateau_delta: float = 0.002,
) -> pd.DataFrame:
    """Detect anomalies across a trajectory of checkpoints.

    Parameters
    ----------
    traj_df : DataFrame with columns [epoch, global_relative_change, checkpoint]
    spike_z : Z-score threshold for sudden drift spike
    plateau_window : consecutive checkpoints with < plateau_delta change → plateau
    plateau_delta : max delta for plateau detection

    Returns a DataFrame of anomaly events.
    """
    if traj_df.empty or "global_relative_change" not in traj_df.columns:
        return pd.DataFrame()

    gdf = (
        traj_df[["checkpoint", "epoch", "global_relative_change"]]
        .drop_duplicates()
        .sort_values("epoch")
        .reset_index(drop=True)
    )

    if len(gdf) < 2:
        return pd.DataFrame()

    vals = gdf["global_relative_change"].values.astype(float)
    deltas = np.diff(vals)
    mean_d, std_d = float(np.nanmean(deltas)), float(np.nanstd(deltas) + 1e-12)

    events: List[Dict] = []

    # Spike detection
    for i, d in enumerate(deltas):
        z = (d - mean_d) / std_d
        if abs(z) > spike_z:
            events.append({
                "epoch": gdf.loc[i + 1, "epoch"],
                "checkpoint": gdf.loc[i + 1, "checkpoint"],
                "event": "drift_spike" if d > 0 else "drift_collapse",
                "z_score": round(z, 3),
                "global_relative_change": round(vals[i + 1], 6),
            })

    # Plateau detection
    w = plateau_window
    for i in range(len(vals) - w):
        window = vals[i:i + w + 1]
        if np.ptp(window) < plateau_delta:  # peak-to-peak
            events.append({
                "epoch": gdf.loc[i + w, "epoch"],
                "checkpoint": gdf.loc[i + w, "checkpoint"],
                "event": "plateau",
                "z_score": None,
                "global_relative_change": round(float(np.mean(window)), 6),
            })
            break  # report once per plateau region

    return pd.DataFrame(events) if events else pd.DataFrame()


# ────────────────────────────────────────────────────────────────────────────
# Layer health summary
# ────────────────────────────────────────────────────────────────────────────

def layer_health_summary(df: pd.DataFrame) -> Dict[str, object]:
    """Return a concise health summary dict from a layer comparison DataFrame."""
    if df.empty:
        return {}

    rc = df["relative_change"].dropna()
    cos = df["cosine_similarity"].dropna() if "cosine_similarity" in df.columns else pd.Series(dtype=float)

    n_frozen = int((rc < 1e-6).sum())
    n_exploding = int((rc > 5.0).sum())
    n_cos_collapse = int((cos < 0.5).sum()) if not cos.empty else 0

    top_layer = df.loc[df["relative_change"].idxmax(), "layer"] if not rc.empty else "N/A"

    return {
        "frozen_layers": n_frozen,
        "exploding_layers": n_exploding,
        "cosine_collapse_layers": n_cos_collapse,
        "most_changed_layer": top_layer,
        "mean_relative_change": round(float(rc.mean()), 6) if not rc.empty else float("nan"),
        "median_relative_change": round(float(rc.median()), 6) if not rc.empty else float("nan"),
        "p95_relative_change": round(float(rc.quantile(0.95)), 6) if not rc.empty else float("nan"),
    }
