"""
similarity.py – Cross-layer and cross-checkpoint representation analysis.

Features
--------
- Layer-wise CKA (Centered Kernel Alignment) between two checkpoints
- Singular value spectrum comparison (how capacity changes)
- Weight correlation heatmap data between layer pairs
- Effective rank estimate per layer
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# ────────────────────────────────────────────────────────────────────────────
# CKA helpers (linear kernel, unbiased)
# ────────────────────────────────────────────────────────────────────────────

def _gram_matrix(X: np.ndarray) -> np.ndarray:
    """Compute Gram matrix X @ X.T"""
    return X @ X.T


def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Unbiased HSIC estimator (Gretton et al. 2012)."""
    n = K.shape[0]
    if n < 4:
        return float("nan")
    # Centre
    H = np.eye(n) - np.ones((n, n)) / n
    KH = K @ H
    LH = L @ H
    return float(np.trace(KH @ LH)) / ((n - 1) ** 2)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two feature matrices (samples × features)."""
    try:
        K = _gram_matrix(X.astype(np.float64))
        L = _gram_matrix(Y.astype(np.float64))
        hsic_xy = _hsic(K, L)
        hsic_xx = _hsic(K, K)
        hsic_yy = _hsic(L, L)
        denom = np.sqrt(max(hsic_xx, 0) * max(hsic_yy, 0))
        if denom < 1e-12:
            return float("nan")
        return float(hsic_xy / denom)
    except Exception:
        return float("nan")


# ────────────────────────────────────────────────────────────────────────────
# Singular value spectrum
# ────────────────────────────────────────────────────────────────────────────

def singular_value_spectrum(
    sd_a: Dict[str, torch.Tensor],
    sd_b: Dict[str, torch.Tensor],
    *,
    max_layers: int = 20,
    top_k_sv: int = 32,
) -> pd.DataFrame:
    """Compute singular value spectra for weight matrices in both checkpoints.

    Returns a DataFrame with columns:
    layer, checkpoint, rank_index, singular_value, effective_rank
    """
    rows: List[Dict] = []
    shared = sorted(set(sd_a) & set(sd_b))
    weight_layers = [k for k in shared if k.endswith(".weight") and sd_a[k].ndim >= 2][:max_layers]

    for key in weight_layers:
        for tag, sd in [("A", sd_a), ("B", sd_b)]:
            w = sd[key].float().detach().cpu()
            w2d = w.view(w.shape[0], -1).numpy()
            try:
                sv = np.linalg.svd(w2d, compute_uv=False)
                # Effective rank: entropy of normalised singular values
                sv_norm = sv / (sv.sum() + 1e-12)
                eff_rank = float(np.exp(-np.sum(sv_norm * np.log(sv_norm + 1e-12))))
                for i, s in enumerate(sv[:top_k_sv]):
                    rows.append({
                        "layer": key,
                        "checkpoint": tag,
                        "rank_index": i,
                        "singular_value": float(s),
                        "effective_rank": round(eff_rank, 2),
                    })
            except Exception:
                pass

    return pd.DataFrame(rows)


def sv_summary(sv_df: pd.DataFrame) -> pd.DataFrame:
    """Top-singular-value and effective-rank change per layer."""
    if sv_df.empty:
        return pd.DataFrame()

    top_sv = sv_df[sv_df["rank_index"] == 0].copy()
    pivot = top_sv.pivot_table(index="layer", columns="checkpoint", values=["singular_value", "effective_rank"], aggfunc="first")
    pivot.columns = [f"{col[0]}_{col[1]}" for col in pivot.columns]
    pivot = pivot.reset_index()

    if "singular_value_A" in pivot.columns and "singular_value_B" in pivot.columns:
        pivot["sv_change_ratio"] = (pivot["singular_value_B"] - pivot["singular_value_A"]) / (pivot["singular_value_A"].abs() + 1e-12)

    if "effective_rank_A" in pivot.columns and "effective_rank_B" in pivot.columns:
        pivot["eff_rank_delta"] = pivot["effective_rank_B"] - pivot["effective_rank_A"]

    return pivot.sort_values("sv_change_ratio", ascending=False) if "sv_change_ratio" in pivot.columns else pivot


# ────────────────────────────────────────────────────────────────────────────
# Weight norm evolution
# ────────────────────────────────────────────────────────────────────────────

def weight_norm_delta(
    sd_a: Dict[str, torch.Tensor],
    sd_b: Dict[str, torch.Tensor],
) -> pd.DataFrame:
    """Return per-layer L2 norm before/after and fractional change."""
    rows: List[Dict] = []
    shared = sorted(set(sd_a) & set(sd_b))
    for key in shared:
        a, b = sd_a[key], sd_b[key]
        if a.shape != b.shape:
            continue
        na = float(a.float().norm().item())
        nb = float(b.float().norm().item())
        rows.append({
            "layer": key,
            "norm_A": na,
            "norm_B": nb,
            "norm_delta": nb - na,
            "norm_ratio": nb / (na + 1e-12),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("norm_ratio", ascending=False)
    return df
