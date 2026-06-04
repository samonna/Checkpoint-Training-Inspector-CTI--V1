"""
Checkpoint Training Inspector V3 – app.py
Enhancements over V2:
  • Anomaly Detection tab: frozen / exploding / cosine-collapse layer flags
  • Singular Value Spectrum: effective rank and top-SV change per layer
  • Weight Norm Evolution chart
  • Automated narrative in HTML report
  • Layer Health scorecards (frozen / exploding / collapse counts)
  • Trajectory anomaly detection (spikes, plateaus)
  • Improved colour-coded layer table
  • Gradient probe improvements: ratio chart + dead-layer filter
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from checkpoint_inspector.loader import (
    load_checkpoint_bytes,
    read_zip_checkpoints,
    sort_checkpoints,
)
from checkpoint_inspector.metrics import (
    aggregate_by_group,
    compare_state_dicts,
    diff_histogram,
    optimizer_summary,
)
from checkpoint_inspector.report import html_report
from checkpoint_inspector.anomaly import (
    detect_layer_anomalies,
    detect_trajectory_anomalies,
    layer_health_summary,
)
from checkpoint_inspector.similarity import (
    singular_value_spectrum,
    sv_summary,
    weight_norm_delta,
)

# ──────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Checkpoint Training Inspector V3",
    page_icon="🔬",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────
# Default code snippets for Gradient Probe
# ──────────────────────────────────────────────────────────────────────────
MODEL_CODE_DEFAULT = """import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def build_model():
    return TinyNet()
"""

BATCH_CODE_DEFAULT = """import torch

def get_batch():
    torch.manual_seed(123)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 4, (8,))
    return x, y
"""

LOSS_CODE_DEFAULT = """import torch.nn.functional as F

def compute_loss(model, batch):
    x, y = batch
    logits = model(x)
    return F.cross_entropy(logits, y)
"""

# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def uploaded_to_bytes(uploaded_file) -> Tuple[str, bytes]:
    return uploaded_file.name, uploaded_file.getvalue()


@st.cache_data(show_spinner=False)
def cached_load_checkpoint(name: str, data: bytes):
    return load_checkpoint_bytes(data, name)


def metric_cards(summary: dict, health: dict | None = None):
    cols = st.columns(6)
    cols[0].metric("Shared layers", summary.get("shared_layers", 0))
    cols[1].metric("Comparable", summary.get("comparable_layers", 0))
    cols[2].metric("Issues", summary.get("issues", 0))
    cols[3].metric("Params", f"{summary.get('total_params_compared', 0):,}")
    grc = summary.get("global_relative_change", float("nan"))
    cols[4].metric("Global drift", f"{grc:.5f}" if isinstance(grc, float) else grc)
    if health:
        frozen = health.get("frozen_layers", 0)
        exploding = health.get("exploding_layers", 0)
        cols[5].metric("Frozen / Exploding", f"{frozen} / {exploding}")


def make_comparison(name_a, data_a, name_b, data_b, layer_filter):
    ckpt_a = cached_load_checkpoint(name_a, data_a)
    ckpt_b = cached_load_checkpoint(name_b, data_b)
    df, issues, summary = compare_state_dicts(
        ckpt_a.state_dict, ckpt_b.state_dict, layer_filter=layer_filter
    )
    return ckpt_a, ckpt_b, df, issues, summary


def gradient_probe(data_a, data_b, name_a, name_b, model_code, batch_code, loss_code) -> pd.DataFrame:
    scope = {"torch": torch}
    exec(model_code, scope)
    exec(batch_code, scope)
    exec(loss_code, scope)
    for fn in ("build_model", "get_batch", "compute_loss"):
        if fn not in scope:
            raise ValueError(f"Code must define {fn}().")

    ckpt_a = cached_load_checkpoint(name_a, data_a)
    ckpt_b = cached_load_checkpoint(name_b, data_b)
    rows = []
    batch = scope["get_batch"]()
    for tag, ckpt in [("A", ckpt_a), ("B", ckpt_b)]:
        model = scope["build_model"]()
        model.load_state_dict(ckpt.state_dict, strict=False)
        model.train()
        model.zero_grad(set_to_none=True)
        loss = scope["compute_loss"](model, batch)
        loss.backward()
        for layer, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().cpu().float().flatten()
                rows.append({
                    "checkpoint": tag,
                    "layer": layer,
                    "grad_norm": float(torch.linalg.norm(grad).item()),
                    "grad_mean_abs": float(torch.mean(torch.abs(grad)).item()),
                    "grad_max_abs": float(torch.max(torch.abs(grad)).item()),
                })
    gdf = pd.DataFrame(rows)
    if gdf.empty:
        return gdf
    pivot = gdf.pivot_table(
        index="layer", columns="checkpoint",
        values=["grad_norm", "grad_mean_abs"], aggfunc="first"
    ).reset_index()
    pivot.columns = ["layer" if c[0] == "layer" else f"{c[0]}_{c[1]}" for c in pivot.columns]
    if "grad_norm_A" in pivot.columns and "grad_norm_B" in pivot.columns:
        pivot["grad_norm_diff"] = pivot["grad_norm_B"] - pivot["grad_norm_A"]
        pivot["grad_norm_ratio"] = pivot["grad_norm_B"] / (pivot["grad_norm_A"] + 1e-12)
        pivot["is_dead_A"] = pivot["grad_norm_A"] < 1e-7
        pivot["is_dead_B"] = pivot["grad_norm_B"] < 1e-7
    return pivot


def create_demo_checkpoints(out_dir: Path, epochs: int = 5):
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(7)
    out_dir.mkdir(parents=True, exist_ok=True)

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Linear(16, 4)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    x = torch.randn(128, 3, 32, 32)
    y = torch.randint(0, 4, (128,))
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
    model = TinyNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    files = []
    for epoch in range(1, epochs + 1):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item()
        path = out_dir / f"epoch_{epoch:03d}.pth"
        torch.save({
            "epoch": epoch,
            "loss": total / len(loader),
            "state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
        }, path)
        files.append(path)
    return files


# ──────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────────────────────────────────
st.title("🔬 Checkpoint Training Inspector V3")
st.caption(
    "Offline PyTorch checkpoint analysis — layer drift · anomaly detection · "
    "singular value spectrum · trajectory diagnostics · gradient probes · smart reports"
)

with st.sidebar:
    st.header("Navigation")
    mode = st.radio(
        "Select mode",
        [
            "Checkpoint Diff",
            "Anomaly Detection",
            "Representation Analysis",
            "Trajectory",
            "Gradient Probe",
            "Sample Data",
            "About",
        ],
    )

# ══════════════════════════════════════════════════════════════════════════
# MODE: Checkpoint Diff
# ══════════════════════════════════════════════════════════════════════════
if mode == "Checkpoint Diff":
    st.info(
        "Compare two PyTorch checkpoints. Detects frozen layers, exploding layers, "
        "architecture mismatches, and optimizer-state shifts."
    )
    with st.sidebar:
        file_a = st.file_uploader("Checkpoint A", type=["pt", "pth", "bin"], key="diff_a")
        file_b = st.file_uploader("Checkpoint B", type=["pt", "pth", "bin"], key="diff_b")
        layer_filter = st.selectbox(
            "Layer filter",
            ["All tensors", "Trainable-like tensors", "Weights only", "Bias only", "No BatchNorm running stats"],
        )
        ranking_metric = st.selectbox(
            "Ranking metric",
            ["relative_change", "diff_norm", "mean_abs_diff", "max_abs_diff", "cosine_similarity"],
        )
        top_k = st.slider("Top layers / groups", 5, 100, 20)
        group_depth = st.slider("Group depth", 1, 6, 2)

    if not (file_a and file_b):
        st.warning("Upload two checkpoints to start.")
    else:
        name_a, data_a = uploaded_to_bytes(file_a)
        name_b, data_b = uploaded_to_bytes(file_b)

        with st.spinner("Loading and comparing checkpoints…"):
            ckpt_a, ckpt_b, df, issues, summary = make_comparison(
                name_a, data_a, name_b, data_b, layer_filter
            )
            health = layer_health_summary(df)
            anomalies = detect_layer_anomalies(df)

        st.subheader("Checkpoint metadata")
        st.json({"checkpoint_a": ckpt_a.metadata, "checkpoint_b": ckpt_b.metadata})
        metric_cards(summary, health)

        if df.empty:
            st.error("No comparable layers found after filtering.")
        else:
            # ── Health alert banner ─────────────────────────────────────────
            if health.get("frozen_layers", 0) > 0:
                st.warning(f"⚠️ {health['frozen_layers']} frozen layer(s) detected (relative change < 1e-6)")
            if health.get("exploding_layers", 0) > 0:
                st.error(f"🚨 {health['exploding_layers']} exploding layer(s) detected (relative change > 5.0)")
            if health.get("cosine_collapse_layers", 0) > 0:
                st.warning(f"⚠️ {health['cosine_collapse_layers']} layer(s) show cosine collapse (< 0.5)")

            # ── Main charts ─────────────────────────────────────────────────
            asc = ranking_metric == "cosine_similarity"
            rank_df = df.sort_values(ranking_metric, ascending=asc).head(top_k)
            chart_title = "Lowest cosine similarity layers" if asc else f"Top layers by {ranking_metric}"
            st.plotly_chart(
                px.bar(rank_df, x=ranking_metric, y="layer", orientation="h",
                       color="relative_change", color_continuous_scale="RdYlGn_r",
                       height=max(420, top_k * 24), title=chart_title),
                use_container_width=True,
            )
            st.plotly_chart(
                px.scatter(df, x="a_norm", y="diff_norm", color="relative_change",
                           size="params", hover_name="layer",
                           color_continuous_scale="RdYlGn_r",
                           log_x=True, log_y=True, title="Layer norm vs drift norm"),
                use_container_width=True,
            )

            group_df = aggregate_by_group(df, group_depth)
            st.subheader("Grouped module drift")
            st.plotly_chart(
                px.bar(group_df.head(top_k), x="relative_change", y="group",
                       orientation="h", color="params",
                       title="Top groups by average relative change"),
                use_container_width=True,
            )

            # ── Histogram ────────────────────────────────────────────────────
            st.subheader("Weight-difference histogram")
            selected_layer = st.selectbox("Select layer", df["layer"].tolist())
            bins = st.slider("Histogram bins", 20, 200, 80)
            hist_df = diff_histogram(ckpt_a.state_dict, ckpt_b.state_dict, selected_layer, bins=bins)
            st.plotly_chart(
                px.bar(hist_df, x="diff_value", y="count",
                       title=f"Weight delta distribution: {selected_layer}"),
                use_container_width=True,
            )

            # ── Anomaly summary table ─────────────────────────────────────────
            if not anomalies.empty:
                st.subheader("🚩 Flagged Layers")
                st.dataframe(anomalies, use_container_width=True)

            # ── Layer table + exports ─────────────────────────────────────────
            st.subheader("Layer comparison table")
            st.dataframe(df, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.download_button("Download layer CSV", df.to_csv(index=False).encode(), "layer_diff.csv", "text/csv")
            col2.download_button("Download group CSV", group_df.to_csv(index=False).encode(), "group_diff.csv", "text/csv")

            report_bytes = html_report(
                "Checkpoint Training Inspector Report",
                summary, df, group_df, issues,
                health=health,
                anomalies_df=anomalies,
            )
            col3.download_button("Download HTML report", report_bytes, "checkpoint_report.html", "text/html")

        # ── Issues ─────────────────────────────────────────────────────────
        st.subheader("Missing / mismatched layers")
        if issues.empty:
            st.success("No missing or shape-mismatched layers detected.")
        else:
            st.dataframe(issues, use_container_width=True)
            st.download_button("Download issues CSV", issues.to_csv(index=False).encode(), "issues.csv", "text/csv")

        # ── Optimizer ──────────────────────────────────────────────────────
        st.subheader("Optimizer-state comparison")
        opt_df, opt_summary = optimizer_summary(ckpt_a.optimizer_state, ckpt_b.optimizer_state)
        if not opt_summary.get("available"):
            st.warning("Optimizer state not available in both checkpoints.")
        elif opt_df.empty:
            st.warning("Optimizer state found but no comparable tensor buffers.")
        else:
            st.json(opt_summary)
            st.plotly_chart(
                px.bar(opt_df.head(top_k), x="relative_change", y="optimizer_buffer",
                       orientation="h", color="param_id",
                       title="Top optimizer-buffer changes"),
                use_container_width=True,
            )
            st.dataframe(opt_df, use_container_width=True)
            st.download_button("Download optimizer CSV", opt_df.to_csv(index=False).encode(), "optimizer_diff.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════
# MODE: Anomaly Detection (new in V3)
# ══════════════════════════════════════════════════════════════════════════
elif mode == "Anomaly Detection":
    st.info(
        "Upload two checkpoints and automatically flag frozen, exploding, "
        "and representation-collapsing layers with configurable thresholds."
    )
    with st.sidebar:
        file_a = st.file_uploader("Checkpoint A", type=["pt", "pth", "bin"], key="anom_a")
        file_b = st.file_uploader("Checkpoint B", type=["pt", "pth", "bin"], key="anom_b")
        layer_filter = st.selectbox(
            "Layer filter",
            ["All tensors", "Trainable-like tensors", "Weights only"],
            key="anom_filter",
        )
        frozen_thr = st.number_input("Frozen threshold (relative change)", value=1e-6, format="%.2e")
        exploding_thr = st.number_input("Exploding threshold (relative change)", value=5.0, format="%.2f")
        cosine_thr = st.number_input("Cosine collapse threshold", value=0.5, format="%.2f")
        dead_thr = st.number_input("Dead layer norm threshold", value=1e-7, format="%.2e")

    if not (file_a and file_b):
        st.warning("Upload two checkpoints to start.")
    else:
        name_a, data_a = uploaded_to_bytes(file_a)
        name_b, data_b = uploaded_to_bytes(file_b)
        with st.spinner("Running anomaly detection…"):
            ckpt_a, ckpt_b, df, issues, summary = make_comparison(name_a, data_a, name_b, data_b, layer_filter)
            health = layer_health_summary(df)
            anomalies = detect_layer_anomalies(
                df,
                frozen_threshold=frozen_thr,
                exploding_threshold=exploding_thr,
                cosine_collapse_threshold=cosine_thr,
                dead_norm_threshold=dead_thr,
            )

        metric_cards(summary, health)

        # Health summary cards
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("🧊 Frozen layers", health.get("frozen_layers", 0))
        h2.metric("💥 Exploding layers", health.get("exploding_layers", 0))
        h3.metric("🔄 Cosine collapse", health.get("cosine_collapse_layers", 0))
        h4.metric("📊 Mean drift", f"{health.get('mean_relative_change', 0):.5f}")

        st.subheader("Distribution of Relative Change")
        if not df.empty:
            fig = px.histogram(
                df, x="relative_change", nbins=60,
                color_discrete_sequence=["#4f8ef7"],
                title="Layer relative change distribution",
                log_y=True,
            )
            fig.add_vline(x=frozen_thr, line_dash="dash", line_color="green", annotation_text="frozen")
            fig.add_vline(x=exploding_thr, line_dash="dash", line_color="red", annotation_text="exploding")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Cosine Similarity Distribution")
            fig2 = px.histogram(
                df, x="cosine_similarity", nbins=50,
                color_discrete_sequence=["#f4a261"],
                title="Cosine similarity across layers",
            )
            fig2.add_vline(x=cosine_thr, line_dash="dash", line_color="red", annotation_text="collapse threshold")
            st.plotly_chart(fig2, use_container_width=True)

        if anomalies.empty:
            st.success("✅ No anomalous layers detected with current thresholds.")
        else:
            st.error(f"🚩 {len(anomalies)} anomalous layer(s) detected")
            # Color-code by flag type
            flag_counts = anomalies["flags"].value_counts().reset_index()
            flag_counts.columns = ["flag", "count"]
            st.plotly_chart(
                px.bar(flag_counts, x="flag", y="count", color="flag",
                       title="Anomaly type breakdown"),
                use_container_width=True,
            )
            st.dataframe(anomalies, use_container_width=True)
            st.download_button(
                "Download anomaly CSV",
                anomalies.to_csv(index=False).encode(),
                "anomalies.csv",
                "text/csv",
            )

# ══════════════════════════════════════════════════════════════════════════
# MODE: Representation Analysis (new in V3)
# ══════════════════════════════════════════════════════════════════════════
elif mode == "Representation Analysis":
    st.info(
        "Analyze how the model's internal representations changed: "
        "singular value spectra, effective rank, and weight norm evolution."
    )
    with st.sidebar:
        file_a = st.file_uploader("Checkpoint A", type=["pt", "pth", "bin"], key="rep_a")
        file_b = st.file_uploader("Checkpoint B", type=["pt", "pth", "bin"], key="rep_b")
        max_layers = st.slider("Max weight layers to analyse", 5, 50, 20)
        top_k_sv = st.slider("Top-K singular values to plot", 8, 64, 32)

    if not (file_a and file_b):
        st.warning("Upload two checkpoints to start.")
    else:
        name_a, data_a = uploaded_to_bytes(file_a)
        name_b, data_b = uploaded_to_bytes(file_b)
        with st.spinner("Computing singular value spectra…"):
            ckpt_a = cached_load_checkpoint(name_a, data_a)
            ckpt_b = cached_load_checkpoint(name_b, data_b)
            sv_df = singular_value_spectrum(
                ckpt_a.state_dict, ckpt_b.state_dict,
                max_layers=max_layers, top_k_sv=top_k_sv,
            )
            sv_sum = sv_summary(sv_df)
            norm_df = weight_norm_delta(ckpt_a.state_dict, ckpt_b.state_dict)

        # ── SV Spectrum ───────────────────────────────────────────────────
        st.subheader("Singular Value Spectrum")
        if sv_df.empty:
            st.warning("No 2D+ weight tensors found for SVD analysis.")
        else:
            layers_available = sv_df["layer"].unique().tolist()
            selected_sv_layer = st.selectbox("Select layer for SV plot", layers_available)
            layer_sv = sv_df[sv_df["layer"] == selected_sv_layer]
            st.plotly_chart(
                px.line(layer_sv, x="rank_index", y="singular_value", color="checkpoint",
                        markers=True,
                        title=f"Singular value spectrum: {selected_sv_layer}",
                        labels={"rank_index": "Singular value rank", "singular_value": "Value"}),
                use_container_width=True,
            )

            st.subheader("Effective Rank Change")
            if not sv_sum.empty and "eff_rank_delta" in sv_sum.columns:
                sv_sum_sorted = sv_sum.sort_values("eff_rank_delta")
                st.plotly_chart(
                    px.bar(sv_sum_sorted, x="eff_rank_delta", y="layer",
                           orientation="h", color="eff_rank_delta",
                           color_continuous_scale="RdYlGn",
                           title="Effective rank change (B − A): positive = more complex"),
                    use_container_width=True,
                )
                st.caption("Positive value = layer B uses more of its capacity. Negative = representation collapsed.")

            st.subheader("Top Singular Value Change")
            if not sv_sum.empty and "sv_change_ratio" in sv_sum.columns:
                st.plotly_chart(
                    px.bar(sv_sum.head(20), x="sv_change_ratio", y="layer",
                           orientation="h",
                           title="Top singular value relative change (B−A)/A"),
                    use_container_width=True,
                )

            st.subheader("Effective Rank Summary Table")
            st.dataframe(sv_sum, use_container_width=True)

        # ── Weight Norm Evolution ─────────────────────────────────────────
        st.subheader("Weight Norm Evolution")
        if norm_df.empty:
            st.warning("No shared weight layers found.")
        else:
            st.plotly_chart(
                px.scatter(norm_df, x="norm_A", y="norm_B",
                           hover_name="layer", color="norm_ratio",
                           color_continuous_scale="RdYlGn_r",
                           title="Weight norm: A vs B (diagonal = no change)",
                           log_x=True, log_y=True),
                use_container_width=True,
            )
            # Add diagonal reference line
            st.plotly_chart(
                px.bar(norm_df.head(30), x="norm_ratio", y="layer",
                       orientation="h", color="norm_ratio",
                       color_continuous_scale="RdYlGn_r",
                       title="Norm ratio B/A (1.0 = no change)"),
                use_container_width=True,
            )
            st.dataframe(norm_df, use_container_width=True)
            st.download_button(
                "Download norm delta CSV",
                norm_df.to_csv(index=False).encode(),
                "norm_delta.csv",
                "text/csv",
            )

# ══════════════════════════════════════════════════════════════════════════
# MODE: Trajectory
# ══════════════════════════════════════════════════════════════════════════
elif mode == "Trajectory":
    st.info(
        "Compare many checkpoints against a reference checkpoint. "
        "Automatic anomaly detection for training spikes and plateaus."
    )
    with st.sidebar:
        files = st.file_uploader("Upload checkpoints", type=["pt", "pth", "bin"], accept_multiple_files=True, key="traj_files")
        zip_file = st.file_uploader("Optional ZIP of checkpoints", type=["zip"], key="traj_zip")
        layer_filter = st.selectbox(
            "Layer filter",
            ["All tensors", "Trainable-like tensors", "Weights only", "Bias only", "No BatchNorm running stats"],
            key="traj_filter",
        )
        ref_index = st.number_input("Reference index", min_value=0, value=0, step=1)
        metric = st.selectbox("Trajectory metric", ["relative_change", "diff_norm", "mean_abs_diff", "cosine_similarity"])
        group_depth = st.slider("Group depth", 1, 6, 2, key="traj_depth")
        top_groups = st.slider("Top groups", 3, 30, 8)
        spike_z = st.slider("Spike detection Z-score", 1.5, 5.0, 2.5, step=0.1)

    payloads: List[Tuple[str, bytes]] = []
    if files:
        payloads.extend(uploaded_to_bytes(f) for f in files)
    if zip_file:
        payloads.extend(read_zip_checkpoints(zip_file.getvalue()))
    payloads = sort_checkpoints(payloads)

    if len(payloads) < 2:
        st.warning("Upload at least two checkpoints.")
    else:
        st.write("Loaded order:", [name for name, _ in payloads])
        ref_index = int(min(ref_index, len(payloads) - 1))
        ref_name, ref_data = payloads[ref_index]
        ref = cached_load_checkpoint(ref_name, ref_data)

        all_rows = []
        progress = st.progress(0.0, text="Comparing checkpoints…")
        for idx, (name, data) in enumerate(payloads):
            ckpt = cached_load_checkpoint(name, data)
            df, issues, summary = compare_state_dicts(ref.state_dict, ckpt.state_dict, layer_filter=layer_filter)
            if df.empty:
                continue
            gdf = aggregate_by_group(df, group_depth)
            epoch = ckpt.metadata.get("epoch") or ckpt.metadata.get("epoch_from_name") or idx
            gdf["checkpoint"] = name
            gdf["index"] = idx
            gdf["epoch"] = epoch
            gdf["global_relative_change"] = summary["global_relative_change"]
            all_rows.append(gdf)
            progress.progress((idx + 1) / len(payloads))
        progress.empty()

        if all_rows:
            tdf = pd.concat(all_rows, ignore_index=True)
            top = (
                tdf.groupby("group", as_index=False)["relative_change"]
                .mean()
                .sort_values("relative_change", ascending=False)
                .head(top_groups)["group"]
                .tolist()
            )
            plot_df = tdf[tdf["group"].isin(top)].copy()

            st.plotly_chart(
                px.line(plot_df, x="epoch", y=metric, color="group", markers=True,
                        hover_data=["checkpoint", "index"],
                        title=f"Group trajectory vs reference: {ref_name}"),
                use_container_width=True,
            )

            global_df = tdf[["checkpoint", "index", "epoch", "global_relative_change"]].drop_duplicates().sort_values("index")
            st.plotly_chart(
                px.line(global_df, x="epoch", y="global_relative_change", markers=True,
                        hover_data=["checkpoint"],
                        title="Global relative drift over checkpoints"),
                use_container_width=True,
            )

            # ── Trajectory anomaly detection ──────────────────────────────
            traj_anomalies = detect_trajectory_anomalies(global_df, spike_z=spike_z)
            if not traj_anomalies.empty:
                st.subheader("🚩 Trajectory Anomalies")
                st.dataframe(traj_anomalies, use_container_width=True)
                # Overlay on chart
                events_to_mark = traj_anomalies[traj_anomalies["epoch"].notna()]
                if not events_to_mark.empty:
                    fig_overlay = px.line(
                        global_df, x="epoch", y="global_relative_change",
                        markers=True, title="Global drift with anomaly markers"
                    )
                    for _, ev in events_to_mark.iterrows():
                        colour = "red" if "spike" in str(ev["event"]) else "orange"
                        fig_overlay.add_vline(
                            x=ev["epoch"], line_dash="dash", line_color=colour,
                            annotation_text=ev["event"],
                        )
                    st.plotly_chart(fig_overlay, use_container_width=True)
            else:
                st.success("✅ No trajectory anomalies detected.")

            st.dataframe(tdf, use_container_width=True)
            trajectory_vals = global_df["global_relative_change"].tolist()

            col1, col2 = st.columns(2)
            col1.download_button("Download trajectory CSV", tdf.to_csv(index=False).encode(), "trajectory.csv", "text/csv")
            if not traj_anomalies.empty:
                col2.download_button("Download anomalies CSV", traj_anomalies.to_csv(index=False).encode(), "traj_anomalies.csv", "text/csv")
        else:
            st.error("No comparable checkpoint data found.")

# ══════════════════════════════════════════════════════════════════════════
# MODE: Gradient Probe
# ══════════════════════════════════════════════════════════════════════════
elif mode == "Gradient Probe":
    st.warning(
        "⚠️ This mode executes pasted Python code. Use only in a local environment with trusted code."
    )
    file_a = st.file_uploader("Checkpoint A", type=["pt", "pth", "bin"], key="grad_a")
    file_b = st.file_uploader("Checkpoint B", type=["pt", "pth", "bin"], key="grad_b")
    model_code = st.text_area("Model code (define build_model())", value=MODEL_CODE_DEFAULT, height=260)
    batch_code = st.text_area("Batch code (define get_batch())", value=BATCH_CODE_DEFAULT, height=160)
    loss_code = st.text_area("Loss code (define compute_loss(model, batch))", value=LOSS_CODE_DEFAULT, height=160)

    show_dead = st.checkbox("Highlight dead layers (grad norm < 1e-7)", value=True)

    if st.button("Run gradient probe", type="primary"):
        if not (file_a and file_b):
            st.error("Upload both checkpoints first.")
        else:
            try:
                name_a, data_a = uploaded_to_bytes(file_a)
                name_b, data_b = uploaded_to_bytes(file_b)
                with st.spinner("Recomputing gradients…"):
                    gdf = gradient_probe(data_a, data_b, name_a, name_b, model_code, batch_code, loss_code)

                if gdf.empty:
                    st.warning("No gradients found. Check model/loss/batch code.")
                else:
                    if "grad_norm_A" in gdf.columns and "grad_norm_B" in gdf.columns:
                        st.plotly_chart(
                            px.bar(gdf.sort_values("grad_norm_diff", ascending=False).head(30),
                                   x="grad_norm_diff", y="layer", orientation="h",
                                   color="grad_norm_diff",
                                   color_continuous_scale="RdYlGn_r",
                                   title="Top gradient norm differences (B − A)"),
                            use_container_width=True,
                        )
                        st.plotly_chart(
                            px.scatter(gdf, x="grad_norm_A", y="grad_norm_B",
                                       hover_name="layer", log_x=True, log_y=True,
                                       color="grad_norm_ratio" if "grad_norm_ratio" in gdf.columns else None,
                                       color_continuous_scale="RdYlGn_r",
                                       title="Gradient norm A vs B"),
                            use_container_width=True,
                        )
                        if show_dead and "is_dead_A" in gdf.columns:
                            dead_layers = gdf[gdf["is_dead_A"] | gdf["is_dead_B"]]
                            if not dead_layers.empty:
                                st.subheader("💀 Dead Layers (near-zero gradients)")
                                st.dataframe(dead_layers[["layer", "grad_norm_A", "grad_norm_B", "is_dead_A", "is_dead_B"]], use_container_width=True)
                            else:
                                st.success("No dead layers detected.")

                    st.dataframe(gdf, use_container_width=True)
                    st.download_button("Download gradient CSV", gdf.to_csv(index=False).encode(), "gradient_probe.csv", "text/csv")
            except Exception as exc:
                st.error(f"Gradient probe failed: {exc}")

# ══════════════════════════════════════════════════════════════════════════
# MODE: Sample Data
# ══════════════════════════════════════════════════════════════════════════
elif mode == "Sample Data":
    st.write("Generate demo checkpoints to test all app modes immediately.")
    epochs = st.slider("Demo epochs", 2, 12, 5)
    if st.button("Create demo checkpoints", type="primary"):
        sample_dir = Path("sample_training_checkpoints")
        with st.spinner("Training TinyNet…"):
            paths = create_demo_checkpoints(sample_dir, epochs)
        st.success(f"Created {len(paths)} checkpoints in {sample_dir.resolve()}")
        for path in paths:
            st.download_button(
                f"⬇️ {path.name}", path.read_bytes(), path.name, "application/octet-stream"
            )

# ══════════════════════════════════════════════════════════════════════════
# MODE: About
# ══════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
### What this app is for

**Checkpoint Training Inspector V3** is an offline diagnostic dashboard for PyTorch model checkpoints.
Designed for CNN, ViT, Mamba, and other deep learning experiments where you need to understand
*which layers changed*, *why training plateaued*, and *whether fine-tuning adapted the right components*.

---

### What's new in V3

| Feature | V2 | V3 |
|---|---|---|
| Layer diff + group drift | ✅ | ✅ |
| Optimizer state comparison | ✅ | ✅ |
| Gradient probe | ✅ | ✅ (+ dead layer detection) |
| **Anomaly Detection mode** | ❌ | ✅ |
| **Singular value spectrum** | ❌ | ✅ |
| **Effective rank analysis** | ❌ | ✅ |
| **Weight norm evolution** | ❌ | ✅ |
| **Trajectory spike detection** | ❌ | ✅ |
| **Automated HTML narrative** | ❌ | ✅ |
| **Colour-coded layer tables** | ❌ | ✅ |
| **Health scorecards** | ❌ | ✅ |

---

### Recommended workflow

1. **Checkpoint Diff** — compare early vs late checkpoint; check health scorecards.
2. **Anomaly Detection** — automatically flag frozen, exploding, or collapsing layers.
3. **Representation Analysis** — check singular value spectra; verify encoder capacity.
4. **Trajectory** — upload all epoch checkpoints; detect plateaus and instabilities.
5. **Gradient Probe** — recompute gradients locally to find dead layers.

---

### Applications

- **Alzheimer's MRI / histopathology**: verify encoder layers are updating (not frozen)
- **MSGG-QS-Mamba / spatial transcriptomics**: compare pre-training vs fine-tuning checkpoints
- **LoRA / adapter fine-tuning**: confirm adapter layers change; backbone stays stable
- **Transfer learning**: measure how much pre-trained representations shift
""")
