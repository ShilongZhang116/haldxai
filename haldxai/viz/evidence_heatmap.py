# -*- coding: utf-8 -*-
"""
haldxai.viz.evidence_heatmap
--------------------------------
Draw a candidate × seed evidence heatmap for HALD.

Key features
- Candidate axis filtering by entity_type / seeds
- Row normalization / global normalization
- Optional log1p on edge weights
- Optional weak-seed-column pruning by strength (sum/mean/max, raw or normalized space)
- Row type strip + column strength bar
- Auto word-wrapping for long seed names
- SVG/PDF export with editable text

Author: HALDxAI
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Mapping
from pathlib import Path
import json

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------------
# Defaults & small utilities
# ---------------------------
# Keep text as text in vector exports
def set_export_fonts() -> None:
    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})

# Default type color map (used by the row type strip)
TYPE_COLORS: Mapping[str, str] = {
    "BMC": "#F9D622",
    "EGR": "#F28D21",
    "ASPKM": "#CC6677",
    "CRD": "#459FC4",
    "APP": "#FF7676",
    "SCN": "#44AA99",
    "AAI": "#117733",
    "CRBC": "#332288",
    "NM": "#AA4499",
    "EF": "#88CCEE",
    "Other": "#CBD5E1",
}

def _minmax01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    mn, mx = float(s.min()), float(s.max())
    return (s - mn) / (mx - mn + 1e-12) if mx > mn else s * 0.0

def _read_ids(p: Path) -> set[str]:
    if not Path(p).exists():
        return set()
    return {x.strip() for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()}

def load_scores_table(
    scores_csv: Path,
    id2name_json: Path | None = None,
    weights_parquet: Path | None = None,
) -> pd.DataFrame:
    """
    Load discovery table and try to complete missing columns (name/entity_type/is_seed).
    Requires: entity_id (str), A_score, L_score (others optional).
    """
    df = pd.read_csv(scores_csv, low_memory=False)
    df["entity_id"] = df["entity_id"].astype(str)

    # name
    if "name" not in df.columns and id2name_json and Path(id2name_json).exists():
        id2name = json.loads(Path(id2name_json).read_text(encoding="utf-8"))
        df["name"] = df["entity_id"].map(lambda x: id2name.get(x, x))
    df["name"] = df.get("name", df["entity_id"])

    # entity_type
    if "entity_type" not in df.columns and weights_parquet and Path(weights_parquet).exists():
        w = pd.read_parquet(weights_parquet, columns=["entity_id", "entity_type"])
        w["entity_id"] = w["entity_id"].astype(str)
        df = df.merge(w.drop_duplicates("entity_id"), on="entity_id", how="left")
    df["entity_type"] = df.get("entity_type", "Other").fillna("Other")

    # is_seed default False (caller may OR with external lists)
    if "is_seed" not in df.columns:
        df["is_seed"] = False

    return df

def _pick_top_seeds(all_ids: list[str], scores: pd.DataFrame, col: str, m: int) -> list[str]:
    """Within all_ids, keep ids present in scores and take top m by scores[col]."""
    sc = scores.set_index("entity_id")
    cand = [x for x in all_ids if x in sc.index]
    if col in sc.columns:
        cand = sorted(cand, key=lambda x: float(sc.at[x, col]), reverse=True)
    return cand[:m]

def _agg_edge_weight(edges: pd.DataFrame, u: str, v: str, how: str = "sum") -> float:
    """Sum or max of w_final for (u,v) (both directions)."""
    sub = edges[((edges["u"] == u) & (edges["v"] == v)) | ((edges["u"] == v) & (edges["v"] == u))]
    if sub.empty:
        return 0.0
    w = pd.to_numeric(sub["w_final"], errors="coerce").fillna(0.0)
    return float(w.sum() if how == "sum" else w.max())

def _wrap(s: str, width: Optional[int]) -> str:
    """Very simple word-wrap by character width (for seed tick labels)."""
    s = str(s)
    if not width or width <= 0 or len(s) <= width:
        return s
    out = []
    while len(s) > width:
        out.append(s[:width])
        s = s[width:]
    if s:
        out.append(s)
    return "\n".join(out)


# ---------------------------
# Public API
# ---------------------------
@dataclass
class EvidenceHeatmapSpec:
    # Required IO
    subnet_dir: Path                  # directory containing edges_with_confidence.csv
    scores_csv: Path                  # discovery table with A/L/B columns
    priors_dir: Path                  # contains: aging_ids.txt, longevity_ids.txt
    outdir: Path                      # output dir for figures

    # Labels (only used in titles/filenames if you choose to use them)
    task_name: str = "HALD"

    # Candidate selection
    axis_for_candidates: Literal["bridge", "aging", "longevity"] = "bridge"
    topk_candidates: int = 20
    topm_aging_seeds: int = 20
    topm_longevity_seeds: int = 20
    include_types: Optional[set[str]] = None
    exclude_types: Optional[set[str]] = None
    drop_seeds: bool = False

    # Aggregation & normalization
    agg: Literal["sum", "max"] = "sum"
    normalize: Literal["row", "global", "none"] = "row"
    log1p_weight: bool = False

    # Visuals
    cmap: str = "viridis"
    show_sidebars: bool = True
    show_row_type_strip: bool = True
    show_col_strength_bar: bool = True
    mark_row_top1: bool = True
    xtick_wrap: int = 18
    ytick_fontsize: float = 12
    xtick_fontsize: float = 10
    fig_size: tuple[float, float] = (14, 8)
    dpi: int = 300
    outbase: Optional[str] = None
    save_fig: bool = True

    # Optional colors for row type strip (fallback to module TYPE_COLORS)
    type_colors: Optional[Mapping[str, str]] = None

    # Visual dampening of very weak signals after normalization
    clip_low: Optional[float] = 0.05

    # Prune weak seed columns (by column strength)
    prune_weak_cols: bool = True
    col_strength_space: Literal["raw", "normalized"] = "raw"
    col_strength_metric: Literal["sum", "mean", "max"] = "sum"
    min_col_strength: Optional[float] = None
    min_col_strength_quantile: Optional[float] = 0.10
    keep_min_per_side: int = 5


def draw_seed_candidate_heatmap(
    spec: EvidenceHeatmapSpec,
    id2name_json: Path | None = None,
    type_weights_parquet: Path | None = None,
) -> plt.Figure:
    """
    Render the candidate × seed evidence heatmap.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # ---------- load ----------
    edges = pd.read_csv(spec.subnet_dir / "edges_with_confidence.csv", low_memory=False)
    need = {"u", "v", "w_final"}
    if not need.issubset(edges.columns):
        missing = need - set(edges.columns)
        raise ValueError(f"edges_with_confidence.csv missing columns: {missing}")

    edges["u"] = edges["u"].astype(str)
    edges["v"] = edges["v"].astype(str)
    edges["w_final"] = pd.to_numeric(edges["w_final"], errors="coerce").fillna(0.0)
    if spec.log1p_weight:
        edges["w_final"] = np.log1p(edges["w_final"])

    scores = load_scores_table(spec.scores_csv, id2name_json, type_weights_parquet)

    # seeds for optional filtering & column partition
    aging_all = list(_read_ids(spec.priors_dir / "aging_ids.txt"))
    longe_all = list(_read_ids(spec.priors_dir / "longevity_ids.txt"))
    seed_ids = set(aging_all) | set(longe_all)

    # is_seed update
    scores["is_seed"] = scores.get("is_seed", False).astype(bool) | scores["entity_id"].astype(str).isin(seed_ids)

    # ---------- candidate pool & topK ----------
    candidate_pool = scores.copy()
    if spec.include_types:
        candidate_pool = candidate_pool[candidate_pool["entity_type"].isin(spec.include_types)]
    if spec.exclude_types:
        candidate_pool = candidate_pool[~candidate_pool["entity_type"].isin(spec.exclude_types)]
    if spec.drop_seeds:
        candidate_pool = candidate_pool[~candidate_pool["is_seed"]]

    axis_map = {"bridge": "B_total", "aging": "A_total", "longevity": "L_total"}
    order_col = axis_map[spec.axis_for_candidates]
    if order_col not in candidate_pool.columns:
        raise ValueError(f"{order_col} not found in {spec.scores_csv.name}")

    cand_df = candidate_pool.sort_values(order_col, ascending=False).head(spec.topk_candidates).copy()
    cand_ids = cand_df["entity_id"].tolist()

    # ---------- seed columns (aging + longevity) ----------
    aging_cols = _pick_top_seeds(aging_all, scores, "A_score", spec.topm_aging_seeds)
    longe_cols = _pick_top_seeds(longe_all, scores, "L_score", spec.topm_longevity_seeds)
    cols = aging_cols + longe_cols
    if not cols:
        raise ValueError("No usable seed columns (aging/longevity ids empty or missing in scores).")

    # ---------- build matrix ----------
    mat = np.zeros((len(cand_ids), len(cols)), dtype=float)
    for i, cid in enumerate(cand_ids):
        for j, sid in enumerate(cols):
            mat[i, j] = _agg_edge_weight(edges, cid, sid, spec.agg)

    # ---------- normalize ----------
    if spec.normalize == "row":
        denom = mat.max(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        mat_plot = mat / denom
    elif spec.normalize == "global":
        mmax = mat.max() or 1.0
        mat_plot = mat / mmax
    else:
        mat_plot = mat.copy()

    # ---------- prune weak seed columns ----------
    if spec.prune_weak_cols:
        def _col_strength(M: np.ndarray, how: str) -> np.ndarray:
            if how == "mean":
                return M.mean(axis=0)
            if how == "max":
                return M.max(axis=0)
            return M.sum(axis=0)

        M_for_metric = mat if spec.col_strength_space == "raw" else mat_plot
        col_strength = _col_strength(M_for_metric, spec.col_strength_metric)
        keep_mask = np.isfinite(col_strength)

        if spec.min_col_strength is not None:
            keep_mask &= (col_strength >= float(spec.min_col_strength))

        if spec.min_col_strength_quantile is not None:
            valid = col_strength[np.isfinite(col_strength)]
            if valid.size > 0:
                qthr = float(np.quantile(valid, spec.min_col_strength_quantile))
                keep_mask &= (col_strength >= qthr)

        aging_set = set(aging_all)
        is_aging = np.array([c in aging_set for c in cols])
        idx_all = np.arange(len(cols))
        keep_idx = idx_all[keep_mask]

        def _ensure_min_side(side_mask: np.ndarray):
            nonlocal keep_idx
            side_all = idx_all[side_mask]
            side_keep = np.intersect1d(keep_idx, side_all, assume_unique=False)
            if len(side_keep) < spec.keep_min_per_side and len(side_all) > 0:
                need = spec.keep_min_per_side - len(side_keep)
                order = side_all[np.argsort(-col_strength[side_all])]
                add = [i for i in order if i not in keep_idx][:max(0, need)]
                keep_idx = np.unique(np.concatenate([keep_idx, np.array(add, dtype=int)]))

        _ensure_min_side(is_aging)
        _ensure_min_side(~is_aging)

        if keep_idx.size == 0:
            keep_idx = np.argsort(-col_strength)[:max(1, 2 * spec.keep_min_per_side)]

        keep_idx = np.sort(keep_idx)
        cols = [cols[i] for i in keep_idx]
        mat = mat[:, keep_idx]
        mat_plot = mat_plot[:, keep_idx]

    # ---------- visual dampening of very weak cells ----------
    if isinstance(spec.clip_low, (int, float)):
        lo = float(spec.clip_low)
        mat_plot = np.where(mat_plot < lo, lo, mat_plot)

    # ---------- reorder rows/cols by overall strength ----------
    row_ord = np.argsort(-mat_plot.sum(axis=1))
    col_ord = np.argsort(-mat_plot.sum(axis=0))
    mat_plot = mat_plot[row_ord][:, col_ord]
    cand_df = cand_df.iloc[row_ord].reset_index(drop=True)
    cols = [cols[k] for k in col_ord]

    # Left-side column count for the vertical divider
    nA = sum(1 for c in cols if c in set(aging_all))

    # ---------- plot ----------
    set_export_fonts()
    fig, ax = plt.subplots(figsize=spec.fig_size, dpi=spec.dpi)
    vmin, vmax = float(mat_plot.min()), float(mat_plot.max() if mat_plot.max() > 0 else 1.0)
    im = ax.imshow(mat_plot, aspect="auto", cmap=spec.cmap, vmin=vmin, vmax=vmax)

    # y labels (candidates)
    ax.set_yticks(np.arange(len(cand_df)))
    ax.set_yticklabels(cand_df["name"], fontsize=spec.ytick_fontsize)

    # x labels (seeds)
    sname = scores.set_index("entity_id")["name"]
    seed_names = [_wrap((sname.get(k) if pd.notna(sname.get(k)) else k), spec.xtick_wrap) for k in cols]
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(seed_names, rotation=90, ha="center", fontsize=spec.xtick_fontsize)

    # A|L divider
    ax.axvline(x=nA - 0.5, color="#94A3B8", lw=1.0)
    ax.set_xlabel("Seeds (Aging | Longevity)")
    ax.set_ylabel("Top candidates")

    # sidebars & strips
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    div = make_axes_locatable(ax)
    last_right_ax = ax

    if spec.show_sidebars:
        ax_metric = div.append_axes("right", size="2.8%", pad=0.20)
        vals = cand_df[order_col].to_numpy()
        vmin_s, vmax_s = float(vals.min()), float(max(vals.max(), 1e-12))
        ax_metric.imshow(vals[:, None], aspect="auto", cmap="Greens", vmin=vmin_s, vmax=vmax_s)
        ax_metric.set_xticks([]); ax_metric.set_yticks([]); ax_metric.set_title(order_col, fontsize=8, pad=2)
        last_right_ax = ax_metric

    if spec.show_row_type_strip:
        ax_type = div.append_axes("right", size="1.3%", pad=0.10)
        types = cand_df["entity_type"].fillna("Other").astype(str).tolist()
        type_colors = spec.type_colors if spec.type_colors is not None else TYPE_COLORS
        strip_colors = [type_colors.get(t, type_colors.get("Other", "#CBD5E1")) for t in types]
        ax_type.imshow(np.arange(len(types))[:, None], aspect="auto",
                       cmap=mpl.colors.ListedColormap(strip_colors))
        ax_type.set_xticks([]); ax_type.set_yticks([]); ax_type.set_title("type", fontsize=8, pad=2)
        last_right_ax = ax_type

    # top column strength bar (over normalized matrix)
    if spec.show_col_strength_bar:
        pos = ax.get_position()
        ax_top = fig.add_axes([pos.x0, pos.y1 + 0.02, pos.width, 0.08])
        col_strength_norm = mat_plot.sum(axis=0)
        xs = np.arange(len(cols))
        ax_top.bar(xs[:nA], col_strength_norm[:nA], color="#64748B", width=0.9)
        ax_top.bar(xs[nA:], col_strength_norm[nA:], color="#94A3B8", width=0.9)
        ax_top.set_xlim(-0.5, len(cols) - 0.5); ax_top.set_xticks([]); ax_top.set_yticks([])
        ax_top.set_title("Seed column strength", fontsize=9, pad=0)

    # colorbar on the far right
    cax = fig.add_axes([last_right_ax.get_position().x1 + 0.02,
                        ax.get_position().y0, 0.015, ax.get_position().height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Normalized evidence strength", rotation=90)

    # save
    if spec.save_fig:
        spec.outdir.mkdir(parents=True, exist_ok=True)
        outbase = spec.outbase or f"Fig_{spec.task_name}_SeedCandidateHeatmap"
        for ext in ("svg", "pdf"):
            fig.savefig(spec.outdir / f"{outbase}.{ext}", bbox_inches="tight", dpi=300)

    return fig


__all__ = [
    "EvidenceHeatmapSpec",
    "draw_seed_candidate_heatmap",
    "load_scores_table",
    "set_export_fonts",
    "TYPE_COLORS",
]
