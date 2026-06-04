from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    if not torch.is_tensor(x):
        raise TypeError("Expected torch.Tensor")
    if not torch.is_floating_point(x) and not torch.is_complex(x):
        x = x.float()
    elif torch.is_complex(x):
        x = x.abs().float()
    else:
        x = x.float()
    return x.detach().cpu().numpy()


def is_selected_layer(name: str, tensor: torch.Tensor, layer_filter: str) -> bool:
    if layer_filter == "All tensors":
        return True
    if layer_filter == "Weights only":
        return name.endswith("weight")
    if layer_filter == "Bias only":
        return name.endswith("bias")
    if layer_filter == "No BatchNorm running stats":
        bad = ("running_mean", "running_var", "num_batches_tracked")
        return not any(x in name for x in bad)
    if layer_filter == "Trainable-like tensors":
        bad = ("running_mean", "running_var", "num_batches_tracked")
        return not any(x in name for x in bad) and torch.is_floating_point(tensor)
    return True


def layer_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float | int]:
    av = tensor_to_numpy(a).ravel().astype(np.float64, copy=False)
    bv = tensor_to_numpy(b).ravel().astype(np.float64, copy=False)
    diff = bv - av
    a_norm = float(np.linalg.norm(av))
    b_norm = float(np.linalg.norm(bv))
    diff_norm = float(np.linalg.norm(diff))
    denom = (a_norm * b_norm) + 1e-12
    cosine = float(np.dot(av, bv) / denom) if av.size else float("nan")
    return {
        "params": int(av.size),
        "a_norm": a_norm,
        "b_norm": b_norm,
        "diff_norm": diff_norm,
        "relative_change": diff_norm / (a_norm + 1e-12),
        "mean_abs_diff": float(np.mean(np.abs(diff))) if diff.size else float("nan"),
        "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else float("nan"),
        "cosine_similarity": cosine,
        "mean_a": float(np.mean(av)) if av.size else float("nan"),
        "mean_b": float(np.mean(bv)) if bv.size else float("nan"),
        "std_a": float(np.std(av)) if av.size else float("nan"),
        "std_b": float(np.std(bv)) if bv.size else float("nan"),
        "zero_fraction_a": float(np.mean(av == 0)) if av.size else float("nan"),
        "zero_fraction_b": float(np.mean(bv == 0)) if bv.size else float("nan"),
    }


def compare_state_dicts(sd1: Dict[str, torch.Tensor], sd2: Dict[str, torch.Tensor], *, layer_filter: str = "All tensors") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    keys1, keys2 = set(sd1), set(sd2)
    shared = sorted(keys1 & keys2)
    rows: List[Dict[str, object]] = []
    issue_rows: List[Dict[str, object]] = []
    total_params, total_diff_sq, total_a_sq = 0, 0.0, 0.0

    for key in sorted(keys1 - keys2):
        issue_rows.append({"layer": key, "status": "missing_in_checkpoint_b", "shape_a": tuple(sd1[key].shape), "shape_b": None})
    for key in sorted(keys2 - keys1):
        issue_rows.append({"layer": key, "status": "missing_in_checkpoint_a", "shape_a": None, "shape_b": tuple(sd2[key].shape)})

    for key in shared:
        a, b = sd1[key], sd2[key]
        if not is_selected_layer(key, a, layer_filter):
            continue
        if tuple(a.shape) != tuple(b.shape):
            issue_rows.append({"layer": key, "status": "shape_mismatch", "shape_a": tuple(a.shape), "shape_b": tuple(b.shape)})
            continue
        try:
            stats = layer_stats(a, b)
            stats.update({"layer": key, "shape": tuple(a.shape), "dtype_a": str(a.dtype), "dtype_b": str(b.dtype), "status": "ok"})
            rows.append(stats)
            total_params += int(stats["params"])
            total_diff_sq += float(stats["diff_norm"]) ** 2
            total_a_sq += float(stats["a_norm"]) ** 2
        except Exception as exc:
            issue_rows.append({"layer": key, "status": f"error: {exc}", "shape_a": tuple(a.shape), "shape_b": tuple(b.shape)})

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("relative_change", ascending=False)
    issues = pd.DataFrame(issue_rows)
    summary = {
        "shared_layers": len(shared),
        "issues": len(issue_rows),
        "comparable_layers": len(rows),
        "total_params_compared": total_params,
        "global_relative_change": float(math.sqrt(total_diff_sq) / (math.sqrt(total_a_sq) + 1e-12)) if total_a_sq else float("nan"),
        "missing_in_a": len(keys2 - keys1),
        "missing_in_b": len(keys1 - keys2),
    }
    return df, issues, summary


def group_prefix(name: str, depth: int) -> str:
    parts = name.split(".")
    return ".".join(parts[:depth]) if len(parts) >= depth else name


def aggregate_by_group(df: pd.DataFrame, depth: int = 2) -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.copy()
    tmp["group"] = tmp["layer"].apply(lambda x: group_prefix(str(x), depth))
    agg = tmp.groupby("group", as_index=False).agg(
        params=("params", "sum"),
        layers=("layer", "count"),
        diff_norm=("diff_norm", "sum"),
        relative_change=("relative_change", "mean"),
        mean_abs_diff=("mean_abs_diff", "mean"),
        max_abs_diff=("max_abs_diff", "max"),
        cosine_similarity=("cosine_similarity", "mean"),
    )
    return agg.sort_values("relative_change", ascending=False)


def diff_histogram(sd1: Dict[str, torch.Tensor], sd2: Dict[str, torch.Tensor], layer: str, bins: int = 80) -> pd.DataFrame:
    a, b = sd1[layer], sd2[layer]
    diff = (tensor_to_numpy(b).ravel() - tensor_to_numpy(a).ravel()).astype(np.float64, copy=False)
    counts, edges = np.histogram(diff, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return pd.DataFrame({"diff_value": centers, "count": counts})


def optimizer_summary(opt_a: Optional[Dict], opt_b: Optional[Dict]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if not opt_a or not opt_b:
        return pd.DataFrame(), {"available": False}
    rows = []
    state_a = opt_a.get("state", {}) if isinstance(opt_a, dict) else {}
    state_b = opt_b.get("state", {}) if isinstance(opt_b, dict) else {}
    common_ids = sorted(set(state_a) & set(state_b), key=lambda x: str(x))
    for pid in common_ids:
        a_state, b_state = state_a[pid], state_b[pid]
        for key in sorted(set(a_state) & set(b_state)):
            av, bv = a_state[key], b_state[key]
            if torch.is_tensor(av) and torch.is_tensor(bv) and tuple(av.shape) == tuple(bv.shape):
                stats = layer_stats(av, bv)
                rows.append({"param_id": str(pid), "optimizer_buffer": key, **stats})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("relative_change", ascending=False)
    return df, {"available": True, "common_optimizer_param_ids": len(common_ids), "rows": len(rows)}
