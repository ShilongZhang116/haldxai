# haldxai/viz/al_scatter.py
# -*- coding: utf-8 -*-
"""
A–L 平面散点图（可复用）
========================
- 读取 S2S 阶段的 scores_all_nodes.csv（必要列：A_score, L_score；可选：B_total、name、entity_type、strength_sub、is_seed）
- 若缺 name / entity_type，会自动从 id2name.json / weights.parquet 补齐
- 支持：类型筛选、别名(按 id / 按 name)、强制标注名单、Top-N 标注策略（按 Bridge 或 A+L）、点大小随某列缩放
- 导出：SVG / PDF，字体以文本保留，便于后期在 AI/Acrobat 编辑
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Set
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke

# 复用 lollipop 模块里现成的工具与默认配色
from .lollipop import (
    DEFAULT_TYPE_COLORS,
    load_scores_table,
    set_export_fonts,
)

# 顶部新增：可选使用 adjustText
try:
    from adjustText import adjust_text
    _HAVE_ADJUSTTEXT = True
except Exception:
    _HAVE_ADJUSTTEXT = False

def _read_ids(p: Path) -> set[str]:
    if not p or not Path(p).exists():
        return set()
    return {s.strip() for s in Path(p).read_text(encoding="utf-8").splitlines() if s.strip()}


def _minmax01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    mn, mx = float(s.min()), float(s.max())
    if mx - mn < 1e-12:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    return (s - mn) / (mx - mn)

# 在文件中加入一个简单的回退排重函数（像素坐标迭代，避免重叠）
def _repel_texts(ax, texts, max_iter: int = 200, pad_px: int = 2):
    """
    纯 matplotlib 的简易排重：若两个文本 bbox 相交，就沿 y 方向相反移动。
    - max_iter: 最大迭代次数
    - pad_px:   每次分离的像素补偿
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    for _ in range(max_iter):
        moved = False
        bboxes = [t.get_window_extent(renderer=renderer).expanded(1.02, 1.10) for t in texts]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if bboxes[i].overlaps(bboxes[j]):
                    # 计算需要分离的像素距离
                    overlap_y = min(bboxes[i].y1, bboxes[j].y1) - max(bboxes[i].y0, bboxes[j].y0)
                    if overlap_y <= 0:
                        continue
                    dy_px = overlap_y / 2.0 + pad_px
                    # 像素→数据坐标（仅 y 方向）
                    y0_data = inv.transform((0, 0))[1]
                    y1_data = inv.transform((0, dy_px))[1]
                    dy_data = (y1_data - y0_data)

                    # 上下相反方向推开
                    yi = texts[i].get_position()[1]
                    yj = texts[j].get_position()[1]
                    if yi <= yj:
                        texts[i].set_y(yi - dy_data)
                        texts[j].set_y(yj + dy_data)
                    else:
                        texts[i].set_y(yi + dy_data)
                        texts[j].set_y(yj - dy_data)
                    moved = True
        if not moved:
            break



@dataclass
class ALScatterSpec:
    # 基础输入
    scores_csv: Path
    outdir: Path
    task_name: str = "COMBINED_SEEDS"

    # 可选映射/补全
    id2name_json: Path | None = None
    weights_parquet: Path | None = None
    priors_dir: Path | None = None        # 若提供，会把 aging_ids.txt / longevity_ids.txt 合并为 is_seed

    # 轴向列（默认 L_score 为 X、A_score 为 Y）
    x_col: str = "L_score"
    y_col: str = "A_score"

    # 点样式
    size_col: str | None = "strength_sub" # None → 固定大小
    size_min: float = 120.0               # 像素^2（散点 s 参数）
    size_max: float = 240.0
    alpha: float = 0.85
    edgecolors: str = "none"

    # 类型与颜色
    type_col: str = "entity_type"
    color_map: Mapping[str, str] | None = None
    include_types: Set[str] | None = None # None → 不筛
    exclude_types: Set[str] | None = None

    # 标注策略
    label_top: int = 25                   # 选出多少个做文本标注
    label_include_seeds: bool = True      # Top 评估是否包含种子
    label_by: str = "bridge"              # "bridge" | "sum"  （B_total优先，或 A_score+L_score）
    alias_id: Mapping[str, str] | None = None     # {entity_id: alias}
    alias_name: Mapping[str, str] | None = None   # {original_name: alias}
    alias_regex: Sequence[tuple[re.Pattern, str]] | None = None
    force_label_ids: Set[str] | None = None
    force_label_names: Set[str] | None = None

    # —— 标签排重相关 ——
    avoid_overlap: bool = True          # 是否尝试避免重叠
    use_adjusttext: bool = True         # 优先使用 adjustText（若已安装）
    show_arrows: bool = True            # 排重后是否给偏移较大的标签加引线
    label_max_iter: int = 200           # 排重最大迭代

    # 文本与导出
    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    outbase: str | None = None            # 文件前缀；None→自动用 Fig_{task}_AL_scatter

    # 视觉常量
    text_color: str = "#0F172A"
    fallback_color: str = "#CBD5E1"


def draw_al_plane_scatter(spec: ALScatterSpec) -> dict:
    """
    绘制 A–L 平面散点图；返回 {fig, ax, df, labels_df, save_paths}
    """
    # ---------- 数据读取与补齐 ----------
    df = load_scores_table(
        scores_csv=spec.scores_csv,
        id2name_json=spec.id2name_json,
        weights_parquet=spec.weights_parquet,
    )
    # 校验必需列
    for c in (spec.x_col, spec.y_col):
        if c not in df.columns:
            raise ValueError(f"缺少列 {c}，请检查 {spec.scores_csv.name}")

    # is_seed：优先用表里现成的；如提供了 priors_dir，则与文本清单取并（更稳）
    if spec.priors_dir:
        aging_ids = _read_ids(Path(spec.priors_dir) / "aging_ids.txt")
        longevity_ids = _read_ids(Path(spec.priors_dir) / "longevity_ids.txt")
        seed_all = aging_ids | longevity_ids
        df["is_seed"] = df["is_seed"].astype(bool) | df["entity_id"].astype(str).isin(seed_all)

    # entity_name：若表里已有 name 列，load_scores_table 已填好；这里做显式别名
    alias_id = dict(spec.alias_id or {})
    alias_name = dict(spec.alias_name or {})
    alias_regex = list(spec.alias_regex or [])

    def _apply_alias(row) -> str:
        eid = str(row["entity_id"])
        if eid in alias_id:
            return alias_id[eid]
        nm = str(row["name"])
        if nm in alias_name:
            return alias_name[nm]
        for pat, rep in alias_regex:
            if pat.search(nm):
                return rep
        return nm

    df["__label_name__"] = df.apply(_apply_alias, axis=1)

    # ---------- 类型筛选与颜色 ----------
    type_col = spec.type_col
    if spec.include_types:
        df = df[df[type_col].isin(spec.include_types)]
    if spec.exclude_types:
        df = df[~df[type_col].isin(spec.exclude_types)]
    cmap = spec.color_map or DEFAULT_TYPE_COLORS
    df["__color__"] = df[type_col].fillna("Other").astype(str).map(cmap).fillna(spec.fallback_color)

    # ---------- 点大小 ----------
    if spec.size_col and spec.size_col in df.columns:
        s01 = _minmax01(df[spec.size_col])
        df["__size__"] = spec.size_min + (spec.size_max - spec.size_min) * s01
    else:
        df["__size__"] = spec.size_min

    # ---------- 选择需要标注的 Top-N ----------
    base = df.copy()
    if not spec.label_include_seeds and "is_seed" in base.columns:
        base = base[~base["is_seed"]]

    if spec.label_by == "bridge" and "B_total" in base.columns:
        lab_df = base.sort_values("B_total", ascending=False).head(spec.label_top).copy()
    else:
        lab_df = base.assign(__tmp__=pd.to_numeric(base.get("A_score", 0), errors="coerce").fillna(0.0) +
                                       pd.to_numeric(base.get("L_score", 0), errors="coerce").fillna(0.0)) \
                     .sort_values("__tmp__", ascending=False).head(spec.label_top).drop(columns="__tmp__")

    # 强制标注并入
    force_ids = set(spec.force_label_ids or set())
    force_names = set(spec.force_label_names or set())
    if force_ids or force_names:
        extra = pd.concat([
            df[df["entity_id"].astype(str).isin(force_ids)],
            df[df["name"].astype(str).isin(force_names)],
        ], axis=0).drop_duplicates(subset=["entity_id"])
        lab_df = pd.concat([lab_df, extra], axis=0).drop_duplicates(subset=["entity_id"])

    # ---------- 绘图 ----------
    set_export_fonts()
    title = spec.title or f"{spec.task_name}: Aging vs Longevity relevance"
    xlabel = spec.xlabel or f"Longevity relevance ({spec.x_col})"
    ylabel = spec.ylabel or f"Aging relevance ({spec.y_col})"
    outbase = spec.outbase or f"Fig_{spec.task_name}_AL_scatter"

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    ax.scatter(df[spec.x_col], df[spec.y_col],
               s=df["__size__"], c=df["__color__"],
               edgecolors=spec.edgecolors, alpha=spec.alpha)

    pts = ax.scatter(df[spec.x_col], df[spec.y_col],
                     s=df["__size__"], c=df["__color__"],
                     edgecolors=spec.edgecolors, alpha=spec.alpha)

    texts = []

    # 文本标注（白描边提升可读性）
    for _, r in lab_df.iterrows():
        t = ax.text(r[spec.x_col], r[spec.y_col], str(r["__label_name__"])[:28],
                    fontsize=9, color=spec.text_color, ha="left", va="bottom",
                    path_effects=[withStroke(linewidth=2, foreground="white")])
        texts.append(t)

    # —— 关键：标签排重 ——
    if spec.avoid_overlap and len(texts) > 1:
        if spec.use_adjusttext and _HAVE_ADJUSTTEXT:
            # 使用 adjustText：自动排布并可加引线
            arrow = dict(arrowstyle="-", lw=0.6, color="#94A3B8", alpha=0.8) if spec.show_arrows else None
            adjust_text(
                texts, ax=ax, add_objects=[pts],
                expand=(1.02, 1.15),
                force_points=(0.05, 0.2),  # 点对文本/文本对文本的“力”
                force_text=(0.2, 0.5),
                lim=spec.label_max_iter,
                arrowprops=arrow
            )
        else:
            # 纯 matplotlib 回退方案
            _repel_texts(ax, texts, max_iter=spec.label_max_iter, pad_px=2)
            if spec.show_arrows:
                # 对位移较大的标签画简易引线
                for t in texts:
                    x_t, y_t = t.get_position()
                    # 以最近的数据点估计“原位置”（即文本初始锚点）
                    ax.plot([x_t, x_t], [y_t, y_t], "-", lw=0.0)  # 仅占位，避免未使用的变量告警
                # 注：简易回退中不强制画线到原点位，避免额外开销；如需，可自行记录原始坐标再绘线
                pass

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, color=spec.text_color)
    ax.grid(True, ls=":", color="#E2E8F0")

    # 导出
    spec.outdir.mkdir(parents=True, exist_ok=True)
    save_svg = spec.outdir / f"{outbase}.svg"
    save_pdf = spec.outdir / f"{outbase}.pdf"
    for ext in ("svg", "pdf"):
        fig.savefig(spec.outdir / f"{outbase}.{ext}", bbox_inches="tight", dpi=300)

    return {
        "fig": fig,
        "ax": ax,
        "df": df,
        "labels_df": lab_df,
        "save_paths": {"svg": save_svg, "pdf": save_pdf},
    }


__all__ = ["ALScatterSpec", "draw_al_plane_scatter"]
