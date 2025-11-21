# File: haldxai/viz/chord.py
# -*- coding: utf-8 -*-
"""
Chord diagram for HALD relationship data, with on-disk caching of the
expensive relation matrix. Designed to live inside the haldxai package.

Usage (Notebook or script):
    from haldxai.viz.chord import run_chord_diagram
    run_chord_diagram()  # or pass your own root/out_dir

Cache:
    <ROOT>/cache/viz/chord/
        - relation_matrix.npy
        - relation_matrix_meta.json
"""

from __future__ import annotations

from pathlib import Path
import json
import logging
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_chord_diagram import chord_diagram

# --------------------------- 全局设置 ---------------------------
mpl.rcParams["svg.fonttype"] = "none"  # SVG 保留文字
mpl.rcParams["pdf.fonttype"] = 42      # PDF 内嵌 TrueType

# 包内默认 ROOT（.../HALDxAI-Project）
_DEFAULT_ROOT = Path(__file__).resolve().parents[2]

# 版本号：用于区分缓存是否需要失效
_VERSION = "0.2.0"

# HALD 类别（如需全局统一，也可迁到单独的 config 模块）
HALD_CLASSES: List[str] = [
    "BMC", "EGR", "ASPKM", "CRD", "APP",
    "SCN", "AAI", "CRBC", "NM", "EF",
]

# 类别 → 固定颜色（键需与 HALD_CLASSES 一致）
COLOR_MAP: dict[str, str] = {
    "BMC"   :   "#F9D622",
    "EGR"   :   "#F28D21",
    "ASPKM" :   "#CC6677",
    "CRD"   :   "#459FC4",
    "APP"   :   "#FF7676",
    "SCN"   :   "#44AA99",
    "AAI"   :   "#117733",
    "CRBC"  :   "#332288",
    "NM"    :   "#AA4499",
    "EF"    :   "#88CCEE",
}

# 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# --------------------------- IO 与缓存 ---------------------------

def _project_paths(root: Path) -> Tuple[Path, Path]:
    """返回实体与关系统一 CSV 路径。"""
    data_dir = root / "data" / "finals"
    ent_csv = data_dir / "all_annotated_entities.csv"
    rel_csv = data_dir / "all_annotated_relationships.csv"
    return ent_csv, rel_csv


def _file_info(p: Path) -> dict:
    st = p.stat()
    return {"path": str(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _cache_dir(root: Path) -> Path:
    d = root / "cache" / "viz" / "chord"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_cached_matrix(root: Path, classes: List[str]) -> np.ndarray | None:
    cdir = _cache_dir(root)
    meta_f = cdir / "relation_matrix_meta.json"
    dat_f = cdir / "relation_matrix.npy"
    if not (meta_f.exists() and dat_f.exists()):
        return None

    try:
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
    except Exception:
        return None

    # 版本或参数变化则失效
    if meta.get("version") != _VERSION:
        return None
    if meta.get("classes") != classes:
        return None

    # 源文件未改变才可复用
    ent_csv, rel_csv = _project_paths(root)
    cur = {"entities": _file_info(ent_csv), "relationships": _file_info(rel_csv)}
    old = meta.get("sources", {})
    for k in ("entities", "relationships"):
        if k not in old:
            return None
        if (old[k].get("size") != cur[k]["size"] or
                old[k].get("mtime_ns") != cur[k]["mtime_ns"]):
            return None

    try:
        matrix = np.load(dat_f)
        return matrix
    except Exception:
        return None


def _save_cache(root: Path, classes: List[str], matrix: np.ndarray) -> None:
    cdir = _cache_dir(root)
    meta_f = cdir / "relation_matrix_meta.json"
    dat_f = cdir / "relation_matrix.npy"

    ent_csv, rel_csv = _project_paths(root)
    meta = {
        "version": _VERSION,
        "classes": classes,
        "sources": {
            "entities": _file_info(ent_csv),
            "relationships": _file_info(rel_csv),
        },
    }
    np.save(dat_f, matrix)
    meta_f.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------- 数据读取与预处理 ---------------------------

def _read_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ent_csv, rel_csv = _project_paths(root)
    entities = pd.read_csv(ent_csv, low_memory=False)
    relationships = pd.read_csv(rel_csv, low_memory=False)
    return entities, relationships


def _clean_pmid(df: pd.DataFrame, col: str = "pmid") -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col])
    df[col] = df[col].astype(int).astype(str)
    return df


# --------------------------- 矩阵构建（可缓存） ---------------------------

def _build_relation_matrix(
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    classes: List[str],
) -> np.ndarray:
    mapping_df = entities[["pmid", "main_text", "entity_type"]].dropna().drop_duplicates()
    pair2type = {
        (row["pmid"], row["main_text"]): row["entity_type"]
        for _, row in mapping_df.iterrows()
    }

    rel = relationships.copy()
    rel["source_key"] = list(zip(rel["pmid"], rel["source_main_text"]))
    rel["target_key"] = list(zip(rel["pmid"], rel["target_main_text"]))
    rel["src_type"] = rel["source_key"].map(pair2type)
    rel["tgt_type"] = rel["target_key"].map(pair2type)

    filtered = rel[
        rel["src_type"].isin(classes) & rel["tgt_type"].isin(classes)
    ].dropna(subset=["src_type", "tgt_type"])

    logging.info("Filtered relationships: %d rows", len(filtered))

    mat = pd.DataFrame(0, index=classes, columns=classes, dtype=int)
    for _, row in filtered.iterrows():
        mat.loc[row["src_type"], row["tgt_type"]] += 1

    return mat.values.astype(int)


def compute_relation_matrix(
    *,
    root: Path | None = None,
    classes: List[str] = HALD_CLASSES,
    force_recompute: bool = False,
    save_csv: bool = False,
) -> np.ndarray:
    """
    计算或从缓存加载关系矩阵。
    - force_recompute=True：无视缓存强制重算
    - save_csv=True：同时把矩阵保存为 CSV（便于审阅）
    """
    root = Path(root) if root is not None else _DEFAULT_ROOT

    if not force_recompute:
        cached = _load_cached_matrix(root, classes)
        if cached is not None:
            logging.info("Loaded relation matrix from cache.")
            return cached

    entities_df, relations_df = _read_data(root)
    entities_df = _clean_pmid(entities_df)
    relations_df = _clean_pmid(relations_df)

    matrix = _build_relation_matrix(entities_df, relations_df, classes)

    # 保存缓存
    _save_cache(root, classes, matrix)

    # 可选：导出 CSV
    if save_csv:
        cache_csv = _cache_dir(root) / "relation_matrix.csv"
        pd.DataFrame(matrix, index=classes, columns=classes).to_csv(cache_csv, encoding="utf-8-sig")
        logging.info("Matrix CSV saved to %s", cache_csv)

    return matrix


# --------------------------- 绘图 ---------------------------

def _colors_for_classes(classes: List[str], color_map: dict[str, str]) -> List[str]:
    missing = [c for c in classes if c not in color_map]
    if missing:
        raise KeyError(f"COLOR_MAP 缺少颜色定义: {missing}")
    return [color_map[c] for c in classes]


def plot_chord(
    classes: List[str],
    relation_counts: np.ndarray,
    *,
    color_map: dict[str, str] = COLOR_MAP,
    title: str | None = "HALD Entity-Class Relationships",
    chord_width: float = 0.6,
    pad: float = 2.0,
    sort: str = "distance",
    show_legend: bool = True,
    figsize: tuple[float, float] = (8, 8),
    dpi: int = 300,
    out_dir: Path | None = None,
    out_basename: str = "fig2-c",
) -> tuple[Path, Path]:
    """
    绘制 chord diagram 并保存为 SVG/PDF。
    返回 (svg_path, pdf_path)。
    """
    matrix = np.asarray(relation_counts)
    N = len(classes)
    assert matrix.shape == (N, N), f"Matrix must be {N}×{N}"

    node_colors = _colors_for_classes(classes, color_map)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    chord_diagram(
        matrix,
        names=classes,
        ax=ax,
        sort=sort,
        pad=pad,
        chordwidth=chord_width,
        colors=node_colors,  # 固定每个类的颜色
    )
    ax.set_title(title, fontsize=14, weight="bold")

    if show_legend:
        handles = [
            mpl.patches.Patch(facecolor=color_map[c], edgecolor="none", label=c)
            for c in classes
        ]
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=min(5, N),
            frameon=False,
            fontsize=9,
            handlelength=1.2,
            columnspacing=1.2,
        )

    plt.tight_layout()

    # 输出目录（默认放到 notebooks/Step8-Article_Results/figs）
    if out_dir is None:
        out_dir = _DEFAULT_ROOT / "notebooks" / "Step8-Article_Results" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    svg_path = out_dir / f"{out_basename}.svg"
    pdf_path = out_dir / f"{out_basename}.pdf"

    fig.savefig(svg_path, format="svg")
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)

    logging.info("Saved figure → %s ; %s", svg_path, pdf_path)
    return svg_path, pdf_path


# --------------------------- 一键执行 ---------------------------

def run_chord_diagram(
    *,
    root: Path | None = None,
    classes: List[str] = HALD_CLASSES,
    force_recompute: bool = False,
    out_dir: Path | None = None,
    out_basename: str = "fig2-c",
    save_matrix_csv: bool = False,
) -> tuple[Path, Path]:
    """
    高层封装：
      1) 计算/读取缓存的关系矩阵
      2) 绘图并保存
    """
    root = Path(root) if root is not None else _DEFAULT_ROOT
    matrix = compute_relation_matrix(
        root=root,
        classes=classes,
        force_recompute=force_recompute,
        save_csv=save_matrix_csv,
    )
    return plot_chord(
        classes=classes,
        relation_counts=matrix,
        out_dir=out_dir,
        out_basename=out_basename,
    )
