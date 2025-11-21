# -*- coding: utf-8 -*-
"""
haldxai.viz.balance_quadrant
----------------------------
A–L bias vs strength scatter (Δ = A − L), with flexible labeling and
optional overlap avoidance via adjustText.

Author: HALDxAI
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Mapping, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Optional: automatic label overlap reduction
try:
    from adjustText import adjust_text
    _HAVE_ADJUSTTEXT = True
except Exception:  # pragma: no cover
    _HAVE_ADJUSTTEXT = False

# ---------------------------
# Small utils & defaults
# ---------------------------
def set_export_fonts() -> None:
    """Keep texts as editable text in SVG/PDF."""
    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})

def _minmax01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    mn, mx = float(s.min()), float(s.max())
    return (s - mn) / (mx - mn + 1e-12) if mx > mn else s * 0.0

def load_scores_table(
    scores_csv: Path,
    id2name_json: Path | None = None,
    weights_parquet: Path | None = None,
) -> pd.DataFrame:
    """
    Load discovery table and try to complete missing columns (name/entity_type/is_seed).
    Requires at least: entity_id, A_score, L_score.
    """
    import json
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

    if "is_seed" not in df.columns:
        df["is_seed"] = False
    return df

# Default color map for entity_type
TYPE_COLORS: Mapping[str, str] = {
    "BMC": "#F9D622", "EGR": "#F28D21", "ASPKM": "#CC6677", "CRD": "#459FC4", "APP": "#FF7676",
    "SCN": "#44AA99", "AAI": "#117733", "CRBC": "#332288", "NM": "#AA4499", "EF": "#88CCEE",
    "Other": "#CBD5E1",
}

# ---------------------------
# Public API
# ---------------------------
@dataclass
class BalanceQuadrantSpec:
    # Required IO
    scores_csv: Path
    outdir: Path
    task_name: str = "HALD"

    # Columns
    a_col: str = "A_score"
    l_col: str = "L_score"
    strength_mode: str = "maxAL"                # "maxAL" | "B_total"
    size_col: Optional[str] = "strength_sub"
    type_col: str = "entity_type"

    # Visual mapping
    color_map: Mapping[str, str] = field(default_factory=lambda: dict(TYPE_COLORS))
    size_min: float = 60.0
    size_max: float = 220.0
    alpha: float = 0.9
    edgecolor: str = "#2F3B53"
    linewidth: float = 0.3

    # Filtering by type
    include_types: Optional[set[str]] = None
    exclude_types: Optional[set[str]] = None

    # Label selection (conditions combine with AND unless otherwise noted)
    label_by: str = "B_total"                    # used only if label_top is not None
    label_top: Optional[int] = 25                # None -> do NOT top-k; label all after filters
    label_delta_range: Optional[Tuple[float, float]] = None
    label_strength_range: Optional[Tuple[float, float]] = None
    label_abs_delta_ge: Optional[float] = None
    label_query: Optional[str] = None            # pandas .eval expression (can use delta/strength/abs_delta)
    force_label_ids: Optional[set[str]] = None
    force_label_names: Optional[set[str]] = None

    # Plot style
    x_eps: float = 0.03
    figsize: tuple[float, float] = (8.2, 6.8)
    dpi: int = 220
    outbase: Optional[str] = None
    save_fig: bool = True

    # Overlap-avoidance options
    avoid_overlap: bool = True
    label_fontsize: float = 8.5
    label_truncate: int = 26
    # adjustText parameters (only used if installed)
    at_expand: Tuple[float, float] = (1.02, 1.10)
    at_force_text: Tuple[float, float] = (0.2, 0.6)
    at_lim: int = 300
    # If you want connector lines:
    draw_connectors: bool = False


def draw_balance_quadrant(
    spec: BalanceQuadrantSpec,
    id2name_json: Path | None = None,
    type_weights_parquet: Path | None = None,
) -> plt.Figure:
    """
    Render the A–L bias vs strength figure and return the Matplotlib Figure.
    """
    # ---- data ----
    df = load_scores_table(spec.scores_csv, id2name_json, type_weights_parquet)
    for c in (spec.a_col, spec.l_col):
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {spec.scores_csv.name}")

    if spec.include_types:
        df = df[df[spec.type_col].isin(spec.include_types)]
    if spec.exclude_types:
        df = df[~df[spec.type_col].isin(spec.exclude_types)]

    A = pd.to_numeric(df[spec.a_col], errors="coerce").fillna(0.0)
    L = pd.to_numeric(df[spec.l_col], errors="coerce").fillna(0.0)
    delta = A - L

    if spec.strength_mode == "B_total" and "B_total" in df.columns:
        strength = pd.to_numeric(df["B_total"], errors="coerce").fillna(0.0)
        strength_name = "B_total"
    else:
        strength = pd.concat([A, L], axis=1).max(axis=1)
        strength_name = "max(A,L)"

    if spec.size_col and spec.size_col in df.columns:
        sizes = spec.size_min + (spec.size_max - spec.size_min) * _minmax01(df[spec.size_col])
    else:
        sizes = pd.Series(spec.size_min, index=df.index)
    colors = df[spec.type_col].map(lambda t: spec.color_map.get(t, spec.color_map.get("Other", "#CBD5E1")))

    # ---- label pool ----
    dfp = df.copy()
    dfp["delta"] = delta
    dfp["strength"] = strength
    dfp["abs_delta"] = delta.abs()

    mask = pd.Series(True, index=dfp.index)
    if spec.label_delta_range is not None:
        lo, hi = spec.label_delta_range
        mask &= dfp["delta"].between(lo, hi)
    if spec.label_strength_range is not None:
        lo, hi = spec.label_strength_range
        mask &= dfp["strength"].between(lo, hi)
    if spec.label_abs_delta_ge is not None:
        mask &= (dfp["abs_delta"] >= float(spec.label_abs_delta_ge))
    if spec.label_query:
        mask &= dfp.eval(spec.label_query)

    pool = dfp[mask]
    order_col = spec.label_by if spec.label_by in dfp.columns else "strength"
    lab = pool.sort_values(order_col, ascending=False).head(spec.label_top) if spec.label_top is not None else pool

    if spec.force_label_ids:
        lab = pd.concat([lab, dfp[dfp["entity_id"].isin(spec.force_label_ids)]], axis=0)
    if spec.force_label_names:
        if "name" not in dfp.columns:
            dfp["name"] = dfp["entity_id"]
        lab = pd.concat([lab, dfp[dfp["name"].isin(spec.force_label_names)]], axis=0)

    lab = lab.drop_duplicates(subset=["entity_id"])

    # ---- plot ----
    set_export_fonts()
    fig, ax = plt.subplots(figsize=spec.figsize, dpi=spec.dpi)

    ax.scatter(delta, strength, s=sizes, c=colors, alpha=spec.alpha,
               ec=spec.edgecolor, lw=spec.linewidth)

    # guides
    ax.axvline(0.0, color="#94A3B8", lw=1.2)
    ax.axvspan(-spec.x_eps, spec.x_eps, color="#94A3B8", alpha=0.12, label="balance band")
    ax.axhline(float(strength.quantile(0.75)), color="#94A3B8", lw=1.0, ls="--")

    # labels
    texts = []
    for _, r in lab.iterrows():
        d = float(r[spec.a_col] - r[spec.l_col])
        if strength_name in r:
            s_val = float(r[strength_name])
        else:
            s_val = float(max(r.get(spec.a_col, 0), r.get(spec.l_col, 0)))
        label = str(r.get("name") or r["entity_id"])
        if spec.label_truncate and len(label) > spec.label_truncate:
            label = label[:spec.label_truncate]
        txt = ax.text(d, s_val, label, fontsize=spec.label_fontsize, ha="center", va="bottom")
        texts.append(txt)

    if spec.avoid_overlap and texts:
        if _HAVE_ADJUSTTEXT:
            # If you want connector lines, set draw_connectors=True
            kwargs = {}
            if spec.draw_connectors:
                kwargs["arrowprops"] = dict(arrowstyle="-", color="#64748B", lw=0.6, shrinkA=6, shrinkB=6)
            try:
                adjust_text(
                    texts, ax=ax,
                    expand=spec.at_expand,
                    force_text=spec.at_force_text,
                    lim=spec.at_lim,
                    only_move={"text": "xy"},
                    **kwargs
                )
            except Exception:
                pass
        # else: silently fall back to naive labels

    ax.set_xlabel(r"$\Delta = A - L$  (positive: aging-leaning)")
    ax.set_ylabel(f"Strength ({strength_name})")
    ax.set_title(f"{spec.task_name}: A–L bias vs strength", pad=6)
    ax.grid(True, ls=":", color="#E2E8F0")

    # save
    if spec.save_fig:
        spec.outdir.mkdir(parents=True, exist_ok=True)
        outbase = spec.outbase or f"Fig_{spec.task_name}_AL_BiasQuadrant"
        for ext in ("svg", "pdf"):
            fig.savefig(spec.outdir / f"{outbase}.{ext}", bbox_inches="tight", dpi=300)

    return fig


__all__ = ["BalanceQuadrantSpec", "draw_balance_quadrant",
           "TYPE_COLORS", "set_export_fonts", "load_scores_table"]
