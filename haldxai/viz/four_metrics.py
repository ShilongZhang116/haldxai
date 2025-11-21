# -*- coding: utf-8 -*-
"""
haldxai.viz.four_metrics
------------------------------------------------
• 汇总 *_eval.csv → Precision / Recall / F1 / Overlap
• 绘制 2×2 横向柱状图 + 两张 legend
• 输出高分辨率 PDF，便于后期编辑
------------------------------------------------
"""

from __future__ import annotations
import os, re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from matplotlib import patches
import matplotlib as mpl
from pathlib import Path

mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
mpl.rcParams.update({
    "pdf.fonttype": 42,          # TrueType
    "ps.fonttype" : 42,
    "font.family" : "Arial",     # ← 全局字体
    "font.sans-serif": ["Arial"]
})

# ---------- util ----------
def _cmap_palette(n: int, cmap, start=.25, end=.9):
    pos = np.linspace(start, end, n)
    return [cmap(p) for p in pos]


# ---------- util ----------
def _summarize_eval_csv(
    eval_dir   : Path,
    model_alias: dict,
    model_type : dict,
    exclude    : set[str] | None = None,   # ← 新增
) -> pd.DataFrame:
    """
    读取 eval_dir 下所有 *_eval.csv，
    汇总为 DataFrame(index=文件前缀, columns=4 指标 + typ + disp)
    """
    rows = []
    for csv in eval_dir.glob("*.csv"):
        key   = csv.stem
        if exclude and key in exclude:  # ← 过滤
            continue
        disp  = model_alias.get(key, re.sub(r"_eval$", "", key))

        df = pd.read_csv(csv)
        need = {"correct_count", "model_count", "manual_count", "overlap"}
        if not need.issubset(df.columns):
            print(f"⚠️ {csv.name} 缺列 {need-df.columns.keys()}，跳过")
            continue

        cor = df.correct_count.sum()
        mod = df.model_count.sum()
        man = df.manual_count.sum()
        prec = cor / mod if mod else 0
        rec  = cor / man if man else 0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0
        ovl  = df.overlap.mean()

        rows.append({
            "key": key,
            "disp": disp,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Overlap": ovl,
            "typ": model_type.get(key, "LLM"),
        })

    if not rows:                                  # 防止空目录
        raise RuntimeError("❌ eval_dir 中没有可用 *_eval.csv 文件")

    return (
        pd.DataFrame(rows)
        .set_index("key")        # 仍用文件前缀当唯一索引
        .sort_index()
    )


# ---------- main API ----------
def plot_four_metrics(
    eval_dir   : str | Path,
    out_dir    : str | Path,
    model_alias: dict[str, str],
    model_type : dict[str, str],
    abbr_dict  : dict[str, str] | None = None,
    main_name   : str = "four_metrics",
    leg_model  : str = "legend_models",
    leg_id     : str = "legend_identity",
    exclude    : set[str] | None = None,        # ← 新增
) -> None:
    """
    Parameters
    ----------
    eval_dir    : 保存 *eval.csv 的目录
    out_dir     : 图片输出目录
    model_alias : {文件名前缀: 显示名称}
    model_type  : {文件名前缀: 'LLM'/'ML'}
    abbr_dict   : {显示名称: 缩写} (可选，只在柱尾显示)
    """
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True, parents=True)
    if abbr_dict is None: abbr_dict = {}

    dfm = _summarize_eval_csv(Path(eval_dir), model_alias, model_type, exclude)

    cmap = cm["YlGnBu"]
    color_map = {k: c for k, c in zip(dfm.index, _cmap_palette(len(dfm), cmap))}

    metrics = ["Precision", "Recall", "F1", "Overlap"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=120); axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        data = dfm.sort_values(metric, ascending=True)
        y = np.arange(len(data))

        bars = ax.barh(
            y, data[metric],
            color=[color_map[k] for k in data.index],
            edgecolor="k",
            hatch=["//" if data.loc[k, "typ"] == "ML" else "" for k in data.index],
            alpha=.8,
        )

        ax.set_yticks(y, data["disp"], fontsize=12)
        xmax = float(data[metric].max()) * 1.1 or 1.0
        ax.set_xlim(0, xmax)
        ax.set_xlabel(metric, fontsize=11)
        ax.grid(False)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        for bar, mdl in zip(bars, data.index):
            w  = bar.get_width()
            yy = bar.get_y() + bar.get_height() / 2
            ax.text(w + xmax*.01, yy, f"{w:.1%}", ha="left", va="center", fontsize=12)
            if abbr_dict:
                ax.text(-xmax*.02, yy, abbr_dict.get(mdl, ""), ha="right", va="center", fontsize=12)

    fig.suptitle("Model vs Manual – Four Metrics", fontsize=16, y=.94)
    fig.tight_layout(rect=[0, 0, 1, .93])
    for ext in ("pdf", "svg"):             # ❷ 一次保存两格式
        fig.savefig(out_dir / f"{main_name}.{ext}",
                    format=ext, dpi=600, bbox_inches="tight")
    plt.close(fig)

    # ------- legend – models -------
    handles = [
        patches.Patch(
            facecolor=color_map[m],
            edgecolor="k",
            hatch="//" if dfm.loc[m, "typ"] == "ML" else "",
            label=m
        ) for m in dfm.index
    ]
    fig_m, ax_m = plt.subplots(figsize=(12, 1.8)); ax_m.axis("off")
    ax_m.legend(handles, dfm.index,
                ncol=max(2, len(handles)//2),
                loc="center", frameon=False,
                handlelength=1.3, columnspacing=1.3,
                fontsize=9, title="Models")
    for ext in ("pdf", "svg"):             # ❸
        fig_m.savefig(out_dir / f"{leg_model}.{ext}",
                      format=ext, dpi=600, bbox_inches="tight")
    plt.close(fig_m)

    # ------- legend – identity -------
    fig_i, ax_i = plt.subplots(figsize=(3.5, 1.2)); ax_i.axis("off")
    id_handles = [
        patches.Patch(facecolor=cmap(.6), edgecolor='k', label="LLM"),
        patches.Patch(facecolor=cmap(.6), edgecolor='k', label="ML", hatch="//"),
    ]
    ax_i.legend(id_handles, ["LLM", "ML"],
                ncol=2, loc="center", frameon=False,
                handlelength=1.5, columnspacing=1.5,
                fontsize=9, title="Identity")
    for ext in ("pdf", "svg"):             # ❹
        fig_i.savefig(out_dir / f"{leg_id}.{ext}",
                      format=ext, dpi=600, bbox_inches="tight")
    plt.close(fig_i)

    print("✅ 主图:", out_dir / main_name)
    print("✅ 模型 legend:", out_dir / leg_model)
    print("✅ 身份 legend:", out_dir / leg_id)
