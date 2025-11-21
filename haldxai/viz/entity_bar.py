# File: haldxai/viz/entity_bar.py
# -*- coding: utf-8 -*-
"""
Entity count bar chart for HALD, with on-disk caching.

API
---
from haldxai.viz.entity_bar import run_entity_counts

svg_path, pdf_path = run_entity_counts(
    root=None,                 # 项目根（含 data/finals/*.csv），默认取包内推断的根
    classes=None,              # 实体类别列表，默认 HALD_CLASSES
    force_recompute=False,     # True 则忽略缓存，重新计算
    out_dir=None,              # 输出目录，默认 notebooks/Step8-Article_Results/figs
    out_basename="fig2-a",     # 输出文件前缀
    save_counts_csv=True,      # 额外把汇总结果保存为 CSV 便于审阅
)
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, List, Tuple
import json
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker

# --------------------------- 全局设置 ---------------------------
mpl.rcParams["svg.fonttype"] = "none"  # SVG 保留文字
mpl.rcParams["pdf.fonttype"] = 42      # PDF 内嵌 TrueType

# 推断项目根（.../HALDxAI-Project）
_DEFAULT_ROOT = Path(__file__).resolve().parents[2]

_VERSION = "0.2.0"

HALD_CLASSES: Final[list[str]] = [
    "BMC", "EGR", "ASPKM", "CRD", "APP",
    "SCN", "AAI", "CRBC", "NM", "EF",
]

ENTITY_COLOR_MAP: Final[dict[str, str]] = {
    "BMC": "#F9D622",
    "EGR": "#FCEA90",
    "ASPKM": "#F4AB5C",
    "CRD": "#F9D5AD",
    "APP": "#FF7676",
    "SCN": "#FFBABA",
    "AAI": "#7CAB7D",
    "CRBC": "#BDD5BE",
    "NM": "#75B7D1",
    "EF": "#BADBE8",
}

BAR_FIG_DPI: Final[int] = 600

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# --------------------------- 路径与缓存 ---------------------------

def _entities_csv(root: Path) -> Path:
    return root / "data" / "finals" / "all_annotated_entities.csv"


def _file_info(p: Path) -> dict:
    st = p.stat()
    return {"path": str(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _cache_dir(root: Path) -> Path:
    d = root / "cache" / "viz" / "entity_bar"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_cached_counts(root: Path, classes: list[str]) -> pd.DataFrame | None:
    cdir = _cache_dir(root)
    meta_f = cdir / "entity_counts_meta.json"
    csv_f = cdir / "entity_counts.csv"
    if not (meta_f.exists() and csv_f.exists()):
        return None
    try:
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
    except Exception:
        return None
    if meta.get("version") != _VERSION:
        return None
    if meta.get("classes") != classes:
        return None

    ent_csv = _entities_csv(root)
    cur = _file_info(ent_csv)
    old = meta.get("source", {})
    if not old or old.get("size") != cur["size"] or old.get("mtime_ns") != cur["mtime_ns"]:
        return None

    try:
        df = pd.read_csv(csv_f)
        return df
    except Exception:
        return None


def _save_counts_cache(root: Path, classes: list[str], df: pd.DataFrame) -> None:
    cdir = _cache_dir(root)
    meta_f = cdir / "entity_counts_meta.json"
    csv_f = cdir / "entity_counts.csv"

    ent_csv = _entities_csv(root)
    meta = {
        "version": _VERSION,
        "classes": classes,
        "source": _file_info(ent_csv),
    }
    df.to_csv(csv_f, index=False, encoding="utf-8-sig")
    meta_f.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------- 计算与绘图 ---------------------------

def compute_entity_counts(
    *,
    root: Path | None = None,
    classes: list[str] | None = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    返回 DataFrame(columns=['entity_type','count'])，按数量降序。
    """
    root = Path(root) if root is not None else _DEFAULT_ROOT
    classes = classes or HALD_CLASSES

    if not force_recompute:
        cached = _load_cached_counts(root, classes)
        if cached is not None:
            logging.info("Loaded entity counts from cache.")
            return cached

    ent_csv = _entities_csv(root)
    entities = pd.read_csv(ent_csv, low_memory=False)

    counts = (
        entities[entities["entity_type"].isin(classes)]["entity_type"]
        .value_counts()
        .rename_axis("entity_type")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    _save_counts_cache(root, classes, counts)
    return counts


def plot_entity_counts(
    counts: pd.DataFrame,
    *,
    out_dir: Path | None = None,
    out_basename: str = "fig2-a",
) -> tuple[Path, Path]:
    """
    绘制实体数量柱状图（对数 y 轴，自 y=1 起），保存 SVG/PDF。
    返回 (svg_path, pdf_path)。
    """
    # 颜色与顺序对齐
    colors = [ENTITY_COLOR_MAP.get(et, "#CCCCCC") for et in counts["entity_type"]]

    fig, ax = plt.subplots(figsize=(12, 10), dpi=BAR_FIG_DPI)
    bars = ax.bar(
        counts["entity_type"],
        counts["count"],
        color=colors,
        linewidth=0,
        alpha=0.9,
        bottom=1,
        log=True,
    )

    for bar in bars:
        top = bar.get_y() + bar.get_height()
        ax.annotate(
            f"{int(top-1):,}",
            xy=(bar.get_x() + bar.get_width() / 2, top),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_ylim(1, counts["count"].max() * 1.15)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

    ax.grid(False, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=90, fontsize=12)
    plt.tight_layout()

    if out_dir is None:
        out_dir = _DEFAULT_ROOT / "notebooks" / "Step8-Article_Results" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    svg_path = out_dir / f"{out_basename}.svg"
    pdf_path = out_dir / f"{out_basename}.pdf"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    logging.info("Saved entity bar → %s ; %s", svg_path, pdf_path)
    return svg_path, pdf_path

def plot_entity_counts_horizontal(
    counts: pd.DataFrame,
    *,
    out_dir: Path | None = None,
    out_basename: str = "fig2-a-horizontal",
    bar_height: float = 0.35,      # ← 调细靠这个，0.35~0.5 比较合适
    edgecolor: str = "none",
    linewidth: float = 0.0,
    annotate: bool = True,
    put_largest_on_top: bool = True,
) -> tuple[Path, Path]:
    """
    实体数量的『横向柱状图』（barh），x 轴使用对数比例且从 1 开始。
    通过 bar_height 控制“柱体”粗细。
    """
    colors = [ENTITY_COLOR_MAP.get(et, "#CCCCCC") for et in counts["entity_type"]]

    # 类别顺序：可选把最大值放顶部（阅读更自然）
    if put_largest_on_top:
        counts = counts.sort_values("count", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=BAR_FIG_DPI)

    bars = ax.barh(
        y=np.arange(len(counts)),
        width=counts["count"].to_numpy(),
        height=bar_height,                # ← 粗细
        color=colors,
        edgecolor=edgecolor,
        linewidth=linewidth,
        left=1,                           # ← 对数轴从 1 起
    )

    # 类别名在 y 轴
    ax.set_yticks(np.arange(len(counts)))
    ax.set_yticklabels(counts["entity_type"], fontsize=12)

    # x 轴对数 & 范围
    xmax = float(counts["count"].max()) * 1.15
    ax.set_xscale("log")
    ax.set_xlim(1, xmax)

    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    # 可选数值标注（在条右端一点）
    if annotate:
        for rect, v in zip(bars, counts["count"]):
            x = rect.get_x() + rect.get_width()
            y = rect.get_y() + rect.get_height()/2
            ax.annotate(f"{int(v):,}", xy=(x, y),
                        xytext=(4, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=11)

    # 美化
    ax.grid(False, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    # 输出
    if out_dir is None:
        out_dir = _DEFAULT_ROOT / "notebooks" / "Step8-Article_Results" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    svg_path = out_dir / f"{out_basename}.svg"
    pdf_path = out_dir / f"{out_basename}.pdf"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    logging.info("Saved entity horizontal bar → %s ; %s", svg_path, pdf_path)
    return svg_path, pdf_path


# === NEW ===
def plot_entity_counts_lollipop(
    counts: pd.DataFrame,
    *,
    out_dir: Path | None = None,
    out_basename: str = "fig2-a-lollipop",
    marker_size: float = 90,
    stem_lw: float = 2.0,
    annotate: bool = True,
) -> tuple[Path, Path]:
    """
    绘制实体数量“棒棒糖图”（对数 y 轴，自 y=1 起），保存 SVG/PDF。
    返回 (svg_path, pdf_path)。
    """
    # 颜色与顺序对齐
    colors = [ENTITY_COLOR_MAP.get(et, "#CCCCCC") for et in counts["entity_type"]]

    fig, ax = plt.subplots(figsize=(12, 10), dpi=BAR_FIG_DPI)

    x = np.arange(len(counts))
    y = counts["count"].to_numpy()

    # stems: 从 1（log 起点）连到 count
    ax.vlines(x=x, ymin=np.ones_like(y), ymax=y, colors=colors, linewidth=stem_lw)

    # “糖球”
    ax.scatter(x, y, s=marker_size, c=colors, edgecolors="black", linewidths=0.6, zorder=3)

    # 注数值（线性格式），放在点上方一点
    if annotate:
        for xi, yi in zip(x, y):
            ax.annotate(f"{int(yi):,}", (xi, yi),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=11)

    # x 轴用类别名
    ax.set_xticks(x)
    ax.set_xticklabels(counts["entity_type"], rotation=90, fontsize=12)

    # 对数 y 轴，自 1 起
    ax.set_ylim(1, y.max() * 1.15)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

    # 美化
    ax.grid(False, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    # 输出
    if out_dir is None:
        out_dir = _DEFAULT_ROOT / "notebooks" / "Step8-Article_Results" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    svg_path = out_dir / f"{out_basename}.svg"
    pdf_path = out_dir / f"{out_basename}.pdf"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    logging.info("Saved entity lollipop → %s ; %s", svg_path, pdf_path)
    return svg_path, pdf_path



def run_entity_counts(
    *,
    root: Path | None = None,
    classes: list[str] | None = None,
    force_recompute: bool = False,
    out_dir: Path | None = None,
    out_basename: str = "fig2-a",
    save_counts_csv: bool = True,
) -> tuple[Path, Path]:
    """
    高层封装：计算/读取缓存 + 绘图保存。
    """
    counts = compute_entity_counts(root=root, classes=classes, force_recompute=force_recompute)

    # 需要时额外导出一个副本到输出目录
    if save_counts_csv:
        cpy_dir = out_dir or (_DEFAULT_ROOT / "notebooks" / "Step8-Article_Results" / "figs")
        cpy_dir.mkdir(parents=True, exist_ok=True)
        counts.to_csv(cpy_dir / f"{out_basename}_counts.csv", index=False, encoding="utf-8-sig")

    return plot_entity_counts(counts, out_dir=out_dir, out_basename=out_basename)
