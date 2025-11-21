# File: haldxai/viz/relation_pies.py
# -*- coding: utf-8 -*-
"""
Relation distribution donut pies for HALD, with on-disk caching.

API
---
from haldxai.viz.relation_pies import run_relation_pies

svg_path, pdf_path = run_relation_pies(
    root=None,                 # 项目根（含 data/finals/*.csv）
    force_recompute=False,     # True 则忽略缓存，重新计算
    out_dir=None,              # 输出目录，默认 notebooks/Step8-Article_Results/figs
    out_basename="fig2-b",     # 输出文件前缀
    preview=True,              # 是否先显示一张带标签的预览图（不保存）
)
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, List
import json
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------- 全局设置 ---------------------------
mpl.rcParams["svg.fonttype"] = "none"  # SVG 保留文字
mpl.rcParams["pdf.fonttype"] = 42      # PDF 内嵌 TrueType

_DEFAULT_ROOT = Path(__file__).resolve().parents[2]
_VERSION = "0.2.0"

# 五大标准关系（顺序用于展示）
STANDARD_RELS: Final[list[str]] = [
    "Causal",
    "Association",
    "Regulatory",
    "Structural/Belonging",
    "Interaction & Feedback",
]

# 颜色映射
RELATION_COLOR_MAP: dict[str, str] = {
    "Causal":                   "#E15759",
    "Association":              "#4E79A7",
    "Regulatory":               "#F28E2B",
    "Interaction & Feedback":   "#59A14F",
    "Structural/Belonging":     "#B07AA1",
    "Unknown":                  "#9D9D9D",
}

# 别名映射（可按需补充）
REL_LISTS: dict[str, list[str]] = {
    "Causal": [
        "Causal", "CAusal", "CAUSAL", "causal", "Causes", "Causality", "Causation",
        "Causal Relationships", "Causal Relationship", "Causative", "Cause", "cause",
        "Association & Causality", "Association & Causal", "Association & Causation",
        "Influence", "Influences", "Influenced by", "Influenced",
        "Affects", "Induces", "Direct", "Inverse", "Outcome",
        "Predictive", "Mediation", "Mediating", "Etiological",
        "Evidence Supporting", "Evidence-based", "Protective", "Treatment",
        "Intervention", "Involved in", "Causative Relationship", "Cause-effect",
    ],
    "Association": [
        "Association", "ASSOCIATION", "Associative", "associative", "Associational",
        "Associated", "ASSOCIATED", "Associated with", "associated with",
        "Associated With", "ASSOC", "Association Relationships",
        "Association Relationship", "Association & Linkage", "Association & Link",
        "Association & Correlation", "Association &",
        "Association & Structural", "Association & Feedback",
        "Association & Interaction", "Connection", "ASSOCIATIVE",
        "No association", "Negative association", "Negative Association",
        "Positive", "Comparison", "Compared", "Correlation",
        "Observational", "Relevant", "Evaluation",
        "Evidence-based", "Trend", "Process", "Descriptive",
        "Exploratory", "No Relationship", "associated",
        "Associative Relationships", "Relationship",
        "Associative Relationship", "Positive association", "Positive Association",
        "association",
    ],
    "Regulatory": [
        "Regulatory", "REGULATORY", "regulatory", "Regulate",
        "Regulatory Relationships", "Regulatory Relationship",
        "Regulates", "REG", "Regulation", "REGulatory",
    ],
    "Structural/Belonging": [
        "Structural/Belonging", "Structural/Belonging Relationships",
        "Structural/Belonging Relationship", "structural/belonging",
        "STRUCTURAL/BELONGING", "structural/Belonging", "Part Relationship",
        "Structural", "structural", "STRUCTURAL", "Structural Relationship",
        "Structural relationship", "structural relationship",
        "Belonging", "BELONGING", "Belonging Relationships",
        "Belongs to", "Part-Whole", "Part of", "Part",
        "component of", "Component", "Subset", "Whole",
        "Instance", "Classification", "Part-of",
        "APP feature", "Structural", "BELONGS_TO", "Structural/Biological",
    ],
    "Interaction & Feedback": [
        "Interaction & Feedback", "INTERACTION & FEEDBACK", "interaction & feedback",
        "Interaction & Feedback Relationships", "Interaction", "Feedback",
        "Association & Feedback", "Association & Interaction",
        "Competitive", "Connected", "INTERACTION &FEEDBACK",
    ],
}
REL_MAP: Final[dict[str, str]] = {alias: std for std, aliases in REL_LISTS.items() for alias in aliases}

LOW_DPI: Final[int] = 120     # 预览
HIGH_DPI: Final[int] = 600    # 发表
SMALL_PCT_THRESHOLD: Final[int] = 5
INNER_RADIUS: Final[float] = 0.4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# --------------------------- 路径与缓存 ---------------------------

def _relations_csv(root: Path) -> Path:
    return root / "data" / "finals" / "all_annotated_relationships.csv"


def _file_info(p: Path) -> dict:
    st = p.stat()
    return {"path": str(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _cache_dir(root: Path) -> Path:
    d = root / "cache" / "viz" / "relation_pies"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_cached_counts(root: Path) -> pd.DataFrame | None:
    cdir = _cache_dir(root)
    meta_f = cdir / "relation_counts_meta.json"
    csv_f = cdir / "relation_counts.csv"
    if not (meta_f.exists() and csv_f.exists()):
        return None
    try:
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
    except Exception:
        return None
    if meta.get("version") != _VERSION:
        return None

    rel_csv = _relations_csv(root)
    cur = _file_info(rel_csv)
    old = meta.get("source", {})
    if not old or old.get("size") != cur["size"] or old.get("mtime_ns") != cur["mtime_ns"]:
        return None

    try:
        df = pd.read_csv(csv_f)
        return df
    except Exception:
        return None


def _save_counts_cache(root: Path, df: pd.DataFrame) -> None:
    cdir = _cache_dir(root)
    meta_f = cdir / "relation_counts_meta.json"
    csv_f = cdir / "relation_counts.csv"

    rel_csv = _relations_csv(root)
    meta = {
        "version": _VERSION,
        "source": _file_info(rel_csv),
        "order": STANDARD_RELS + ["Unknown"],
    }
    df.to_csv(csv_f, index=False, encoding="utf-8-sig")
    meta_f.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------- 计算与绘图 ---------------------------

def _map_standard_relation(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.strip()
    mapped = cleaned.map(REL_MAP)
    return mapped.fillna("Unknown")


def compute_relation_counts(
    *,
    root: Path | None = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    返回 DataFrame(columns=['relation','count','pct'])，按百分比降序。
    """
    root = Path(root) if root is not None else _DEFAULT_ROOT

    if not force_recompute:
        cached = _load_cached_counts(root)
        if cached is not None:
            logging.info("Loaded relation counts from cache.")
            return cached

    rel_csv = _relations_csv(root)
    relations = pd.read_csv(rel_csv, low_memory=False)

    rel_series = _map_standard_relation(relations["relation_type"])
    order = STANDARD_RELS + ["Unknown"]
    counts = (
        rel_series.value_counts()
        .reindex(order, fill_value=0)
        .rename_axis("relation")
        .reset_index(name="count")
    )
    counts["pct"] = counts["count"] / counts["count"].sum() * 100.0
    counts = counts.sort_values("pct", ascending=False).reset_index(drop=True)

    _save_counts_cache(root, counts)
    return counts


def _draw_pie(ax: plt.Axes, counts: pd.DataFrame, *, with_labels: bool) -> None:
    colors = [RELATION_COLOR_MAP.get(r, "#CCCCCC") for r in counts["relation"]]

    wedges, _ = ax.pie(
        counts["count"],
        startangle=90,
        counterclock=False,
        colors=colors,
        radius=1.0,
        wedgeprops=dict(width=1 - INNER_RADIUS, edgecolor="white"),
    )

    if with_labels:
        for i, wedge in enumerate(wedges):
            ang = (wedge.theta2 + wedge.theta1) / 2
            pct = counts.loc[i, "pct"]
            cnt = counts.loc[i, "count"]

            x_cent = np.cos(np.deg2rad(ang)) * (1 + INNER_RADIUS) / 2
            y_cent = np.sin(np.deg2rad(ang)) * (1 + INNER_RADIUS) / 2

            label_txt = f"{pct:.1f}%\n({cnt})"
            if pct < SMALL_PCT_THRESHOLD:
                x_out = np.cos(np.deg2rad(ang)) * 1.15
                y_out = np.sin(np.deg2rad(ang)) * 1.15
                ax.text(x_out, y_out, label_txt, ha="center", va="center", fontsize=9, weight="bold")
                ax.annotate(
                    "",
                    xy=(x_cent * 1.05, y_cent * 1.05),
                    xytext=(x_out * 0.92, y_out * 0.92),
                    arrowprops=dict(arrowstyle="-", lw=0.7),
                )
            else:
                r, g, b, _a = wedge.get_facecolor()
                brightness = r * 0.299 + g * 0.587 + b * 0.114
                text_color = "black" if brightness > 0.6 else "white"
                ax.text(x_cent, y_cent, label_txt, ha="center", va="center", fontsize=10, weight="bold", color=text_color)

            name_x = np.cos(np.deg2rad(ang)) * 0.8
            name_y = np.sin(np.deg2rad(ang)) * 0.8
            ax.text(name_x, name_y, counts.loc[i, "relation"], ha="center", va="center", fontsize=11, weight="bold")

        ax.legend(wedges, counts["relation"], title="Relation Type", bbox_to_anchor=(1.05, 0.95),
                  loc="upper left", fontsize=10)

    ax.axis("equal")


def plot_relation_pies(
    counts: pd.DataFrame,
    *,
    out_dir: Path | None = None,
    out_basename: str = "fig2-b",
    preview: bool = True,
) -> tuple[Path, Path]:
    """
    绘制预览（带标签，不保存）+ 发表图（无文字，保存 SVG/PDF）。
    返回 (svg_path, pdf_path)。
    """
    # 预览
    if preview:
        fig_low, ax_low = plt.subplots(figsize=(10, 8), dpi=LOW_DPI)
        _draw_pie(ax_low, counts, with_labels=True)
        plt.tight_layout()
        plt.show()
        plt.close(fig_low)

    # 发表
    fig_pub, ax_pub = plt.subplots(figsize=(10, 8), dpi=HIGH_DPI)
    _draw_pie(ax_pub, counts, with_labels=False)
    plt.tight_layout()

    if out_dir is None:
        out_dir = _DEFAULT_ROOT / "notebooks" / "Step8-Article_Results" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    svg_path = out_dir / f"{out_basename}.svg"
    pdf_path = out_dir / f"{out_basename}.pdf"
    fig_pub.savefig(svg_path, format="svg", bbox_inches="tight")
    fig_pub.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig_pub)

    logging.info("Saved relation pies → %s ; %s", svg_path, pdf_path)
    return svg_path, pdf_path


def run_relation_pies(
    *,
    root: Path | None = None,
    force_recompute: bool = False,
    out_dir: Path | None = None,
    out_basename: str = "fig2-b",
    preview: bool = True,
    save_counts_csv: bool = True,
) -> tuple[Path, Path]:
    """
    高层封装：计算/读取缓存 + 绘图保存。
    """
    counts = compute_relation_counts(root=root, force_recompute=force_recompute)

    if save_counts_csv:
        cpy_dir = out_dir or (_DEFAULT_ROOT / "notebooks" / "Step8-Article_Results" / "figs")
        cpy_dir.mkdir(parents=True, exist_ok=True)
        counts.to_csv(cpy_dir / f"{out_basename}_counts.csv", index=False, encoding="utf-8-sig")

    return plot_relation_pies(counts, out_dir=out_dir, out_basename=out_basename, preview=preview)
