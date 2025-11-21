# haldxai/viz/lollipop.py
# -*- coding: utf-8 -*-
"""
可复用的棒棒糖图绘制工具（支持 A/L/Bridge 三轴）
================================================
特点：
- 直接从 S2S 阶段生成的 scores_all_nodes.csv 作图；
- 若缺 name / entity_type，自动从 id2name.json 与 weights.parquet 进行补齐；
- 支持 include/exclude 类型筛选、别名映射、剔除种子；
- 导出 SVG/PDF（字体以文本形式保留，便于后期在 AI/Acrobat 编辑）。

典型用法见文件末尾的示例或 Notebook 片段。
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
import re, json, textwrap

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# —— 默认类型配色（可在调用时覆盖）——
DEFAULT_TYPE_COLORS = {
    "BMC":"#F9D622","EGR":"#F28D21","ASPKM":"#CC6677","CRD":"#459FC4","APP":"#FF7676",
    "SCN":"#44AA99","AAI":"#117733","CRBC":"#332288","NM":"#AA4499","EF":"#88CCEE",
    "Disease":"#9C6ADE","Gene":"#2E7D32","Other":"#CBD5E1",
}


def set_export_fonts():
    """设置 Matplotlib 导出为 SVG/PDF 时保留字体为文本。"""
    mpl.rcParams.update({
        "svg.fonttype": "none",  # SVG 不转路径
        "pdf.fonttype": 42,      # PDF Type42
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
    })


def load_scores_table(
    scores_csv: Path,
    id2name_json: Path | None = None,
    weights_parquet: Path | None = None,
) -> pd.DataFrame:
    """
    读取 S2S 阶段的总分表，并尽量补齐 name / entity_type 列。
    返回：DataFrame（至少包含 entity_id, name, entity_type, A_total/L_total/B_total 等列）
    """
    df = pd.read_csv(scores_csv)
    df["entity_id"] = df["entity_id"].astype(str)

    # name 补齐
    if "name" not in df.columns and id2name_json and Path(id2name_json).exists():
        id2name = json.loads(Path(id2name_json).read_text(encoding="utf-8"))
        df["name"] = df["entity_id"].map(lambda x: id2name.get(x, x))

    if "name" not in df.columns:
        df["name"] = df["entity_id"]

    # entity_type 补齐
    if "entity_type" not in df.columns and weights_parquet and Path(weights_parquet).exists():
        w = pd.read_parquet(weights_parquet, columns=["entity_id", "entity_type"])
        w["entity_id"] = w["entity_id"].astype(str)
        df = df.merge(w.drop_duplicates("entity_id"), on="entity_id", how="left")

    if "entity_type" not in df.columns:
        df["entity_type"] = "Other"

    # is_seed 缺省为 False
    if "is_seed" not in df.columns:
        df["is_seed"] = False

    return df


@dataclass
class LollipopSpec:
    """棒棒糖图绘制规格。"""
    score_col: str                # "A_total" / "L_total" / "B_total" / 自定义列
    label_col: str = "name"
    type_col: str = "entity_type"
    quality_col: str = "quality_norm"
    include_types: set[str] | None = None  # 留空→不过滤
    exclude_types: set[str] | None = None
    drop_seeds: bool = True
    alias_by_name: Mapping[str, str] | None = None
    alias_regex: Sequence[tuple[re.Pattern, str]] | None = None
    topn: int = 20
    title: str = "Top candidates"
    outbase: str = "Fig_lollipop_top"
    outdir: Path | None = None
    color_map: Mapping[str, str] = None  # 若为 None 则用默认配色

    def colors(self) -> Mapping[str, str]:
        return self.color_map or DEFAULT_TYPE_COLORS


def lollipop_topmost(df: pd.DataFrame, spec: LollipopSpec) -> None:
    """
    根据 LollipopSpec 在 DataFrame 上绘制“棒棒糖”Top榜，并导出 SVG/PDF。
    """
    d = df.copy()

    # 过滤
    if spec.drop_seeds and "is_seed" in d.columns:
        d = d[~d["is_seed"]]
    if spec.exclude_types:
        d = d[~d[spec.type_col].isin(spec.exclude_types)]
    if spec.include_types:
        d = d[d[spec.type_col].isin(spec.include_types)]

    # 别名映射
    alias_by_name = dict(spec.alias_by_name or {})
    alias_regex = list(spec.alias_regex or [])
    def _alias(s: str) -> str:
        s0 = str(s)
        if s0 in alias_by_name:
            return alias_by_name[s0]
        for pat, rep in alias_regex:
            if pat.search(s0):
                return rep
        return s0
    d["__label__"] = d[spec.label_col].map(_alias)

    # 排序取 TopN
    if spec.score_col not in d.columns:
        raise KeyError(f"列 {spec.score_col} 不存在；可用列示例：{list(d.columns)[:12]} ...")
    d = d.sort_values(spec.score_col, ascending=False).head(spec.topn).reset_index(drop=True)

    # y 轴位置、颜色、大小
    y = np.arange(len(d))
    t = d[spec.type_col].fillna("Other").astype(str)
    colors = [spec.colors().get(tt, spec.colors().get("Other", "#CBD5E1")) for tt in t]

    if spec.quality_col in d.columns:
        s = d[spec.quality_col].fillna(0).clip(0, 1).to_numpy()
        sizes = 70 + 140*s   # 70~210
    else:
        sizes = np.full(len(d), 120.0)

    # 导出参数
    set_export_fonts()

    # 绘制
    fig, ax = plt.subplots(figsize=(6, 6), dpi=220)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E2E8F0", lw=1.0)
    ax.grid(axis="y", color="#F1F5F9", lw=0.7, alpha=0.85)
    for sp in ["top","right","left","bottom"]:
        ax.spines[sp].set_visible(False)

    ax.hlines(y=y, xmin=0, xmax=d[spec.score_col], color="#94A3B8", lw=1.6, alpha=0.9)
    ax.scatter(d[spec.score_col], y, s=sizes, c=colors, ec="#334155", lw=0.4, zorder=3)

    def wrap(s, width=28):
        return "\n".join(textwrap.wrap(str(s), width=width)) if isinstance(s, str) else s

    ax.set_yticks(y)
    ax.set_yticklabels([wrap(s) for s in d["__label__"]])
    ax.invert_yaxis()

    ax.set_xlabel(f"{spec.score_col} (score)")
    ax.set_xlim(0, max(float(d[spec.score_col].max())*1.05, 0.01))
    ax.set_title(spec.title, pad=10)

    # 右侧标注 Top3 数值（可注释）
    for i, r in d.head(3).iterrows():
        ax.text(r[spec.score_col]*1.01, y[i], f"{r[spec.score_col]:.3f}",
                va="center", ha="left", color="#334155", fontsize=9)

    plt.tight_layout()
    outdir = spec.outdir or Path(".")
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        plt.savefig(outdir / f"{spec.outbase}.{ext}", bbox_inches="tight", dpi=300)
    plt.show()


def draw_three_axes_lollipops(
    scores_csv: Path,
    outdir: Path,
    task_name: str,
    *,
    id2name_json: Path | None = None,
    weights_parquet: Path | None = None,
    include_types: set[str] | None = None,
    exclude_types: set[str] | None = None,
    alias_by_name: Mapping[str, str] | None = None,
    alias_regex: Sequence[tuple[re.Pattern, str]] | None = None,
    topn: int = 20,
    color_map: Mapping[str, str] | None = None,
    drop_seeds: bool = True,   # ← 新增：是否剔除种子（默认 True）
) -> None:
    """
    快速在同一套参数下绘制 A_total / L_total / B_total 三张棒棒糖图。
    """
    df = load_scores_table(scores_csv, id2name_json=id2name_json, weights_parquet=weights_parquet)

    spec_common = dict(
        label_col="name",
        type_col="entity_type",
        quality_col="quality_norm",
        include_types=include_types,
        exclude_types=exclude_types,
        drop_seeds=drop_seeds,   # ← 新增：透传
        alias_by_name=alias_by_name,
        alias_regex=alias_regex,
        topn=topn,
        outdir=outdir,
        color_map=color_map or DEFAULT_TYPE_COLORS,
    )

    lollipop_topmost(df, LollipopSpec(
        score_col="A_total",
        title=f"{task_name}: Top Aging candidates",
        outbase=f"Fig_{task_name}_Aging_lollipop_top",
        **spec_common
    ))
    lollipop_topmost(df, LollipopSpec(
        score_col="L_total",
        title=f"{task_name}: Top Longevity candidates",
        outbase=f"Fig_{task_name}_Longevity_lollipop_top",
        **spec_common
    ))
    lollipop_topmost(df, LollipopSpec(
        score_col="B_total",
        title=f"{task_name}: Top Bridge candidates",
        outbase=f"Fig_{task_name}_Bridge_lollipop_top",
        **spec_common
    ))


__all__ = [
    "DEFAULT_TYPE_COLORS",
    "set_export_fonts",
    "load_scores_table",
    "LollipopSpec",
    "lollipop_topmost",
    "draw_three_axes_lollipops",
]
