# -*- coding: utf-8 -*-
"""
网络图（白底 + 比例阴影 + 防重叠 + 重要节点/社群居中）
-------------------------------------------------------------------
- 仅按“坐标跨度比例 + 方向”控制阴影距离（更稳）。
- 节点两层：阴影(Shadow) → 本体(Body)（默认取消白色光晕与描边）。
- 两轮“轻量碰撞分离”减少节点重叠。
- 支持把重要节点/社群尽量放在图中心（多种指标可选）。
- 支持一键保存为 SVG 与 PDF（向量图，文本不转曲线）。
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Tuple, Sequence, Optional

import json
import numpy as np
import pandas as pd
import networkx as nx

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch
import matplotlib.patheffects as pe

# —— 推荐的向量导出设置：文本保留为文字，便于后期编辑 —— #
mpl.rcParams["svg.fonttype"] = "none"   # SVG 用 <text>，不转曲线
mpl.rcParams["pdf.fonttype"] = 42       # PDF 内嵌 TrueType (Type 42)，可选中文字
mpl.rcParams["ps.fonttype"]  = 42       # 如需 EPS/PS
mpl.rcParams["text.usetex"]  = False    # 用 TeX 往往会转曲或产生 Type 3，关掉

# —— 字体：确保中文/英文都能被嵌入（你有哪种就写哪种）——
mpl.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial"
]
mpl.rcParams["axes.unicode_minus"] = False   # 解决负号乱码

# 关系类型默认配色
DEFAULT_REL_COLORS = {
    "Causal":                   "#E15759",
    "Association":              "#4E79A7",
    "Regulatory":               "#F28E2B",
    "Interaction & Feedback":   "#59A14F",
    "Structural/Belonging":     "#B07AA1",
    "Unknown":                  "#9D9D9D",
}
EDGE_GREY = "#A3ACB9"

REL_ALPHA = {
    "Association": 0.65,
    "Interaction & Feedback": 0.85,
    "Regulatory": 0.95,
    "Causal": 1.00,
    "Structural/Belonging": 0.75,
    "Unknown": 0.70,
}

# ====================== 工具函数 ======================

def _load_id2name(path: Optional[Path]) -> Dict[str, str]:
    if not path or not Path(path).exists(): return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}

def _attach_node_types(nodes_df: pd.DataFrame, types_path: Optional[Path]) -> pd.DataFrame:
    """把 HALD 主类型并入 nodes_df（列名：node_type）。"""
    nd = nodes_df.copy()
    nd["entity_id"] = nd["entity_id"].astype(str)
    if not types_path or not Path(types_path).exists():
        nd["node_type"] = nd.get("node_type", "Other"); return nd
    et = pd.read_parquet(types_path, columns=["entity_id","entity_type","weight_norm"])
    et = et.dropna(subset=["entity_id","entity_type"]).copy()
    et["entity_id"] = et["entity_id"].astype(str)
    et["weight_norm"] = pd.to_numeric(et["weight_norm"], errors="coerce").fillna(0.0)
    et_top = et.sort_values(["entity_id","weight_norm"], ascending=[True,False]).drop_duplicates("entity_id")
    nd = nd.merge(et_top[["entity_id","entity_type"]], on="entity_id", how="left")
    allow = {"BMC","AAI","ASPKM","APP","CRD","SCN","EGR","NM","CRBC","EF"}
    nd["node_type"] = nd["entity_type"].where(nd["entity_type"].isin(allow), "Other")
    return nd

def _standardize_edges(edges: pd.DataFrame, weight_col: str="w_final") -> pd.DataFrame:
    """统一到列：u, v, reltype_top, w_norm（0-1）；无向去重保留最大权重。"""
    df = edges.copy()
    # 端点名兼容
    if not {"u","v"}.issubset(df.columns):
        rename = {}
        if {"source","target"}.issubset(df.columns): rename={"source":"u","target":"v"}
        if {"src","tgt"}.issubset(df.columns): rename={"src":"u","tgt":"v"}
        if rename: df = df.rename(columns=rename)
    df["u"] = df["u"].astype(str); df["v"] = df["v"].astype(str)

    # 关系类型
    if "reltype_top" not in df.columns:
        if "reltype_top_x" in df.columns: df["reltype_top"] = df["reltype_top_x"]
        elif "reltype_top_y" in df.columns: df["reltype_top"] = df["reltype_top_y"]
        else: df["reltype_top"] = ""

    # 权重归一
    if weight_col in df.columns:
        w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).to_numpy()
    elif "conf_norm_q" in df.columns:
        w = pd.to_numeric(df["conf_norm_q"], errors="coerce").fillna(0.0).to_numpy()
    elif "confidence" in df.columns:
        c = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0).to_numpy()
        w = (c - c.min())/(c.max()-c.min()+1e-9)
    else:
        base = pd.to_numeric(df.get("weight", df.get("count", pd.Series(1, index=df.index))),
                             errors="coerce").fillna(0.0).to_numpy()
        w = (base - base.min())/(base.max()-base.min()+1e-9)
    df["w_norm"] = np.clip(w, 0.0, 1.0)

    # 无向规范 + 去重（取最大 w_norm）
    uu = df[["u","v"]].min(axis=1); vv = df[["u","v"]].max(axis=1)
    df["u"], df["v"] = uu, vv
    df = (df.groupby(["u","v"], as_index=False)
            .agg(w_norm=("w_norm","max"),
                 reltype_top=("reltype_top", lambda s: s.value_counts().idxmax() if len(s.dropna()) else "")))
    return df

def _node_sizes(G: nx.Graph, min_size=42, max_size=170) -> np.ndarray:
    """按子网强度（w_vis 累加）决定节点大小；log1p 压缩到 [min,max]。"""
    strength = {n:0.0 for n in G.nodes()}
    for u,v in G.edges():
        wv = float(G[u][v].get("w_vis", G[u][v].get("w",0.0)))
        strength[u]+=wv; strength[v]+=wv
    s = np.array([strength[n] for n in G.nodes()], dtype=float)
    s = np.log1p(s); s = (s - s.min())/(s.max()-s.min()+1e-9)
    return min_size + (max_size-min_size)*s

def _repel_overlaps(pos: Dict[str, np.ndarray],
                    sizes: np.ndarray,
                    keys: Iterable[str],
                    base_sep: float = 0.025,
                    scale_sep: float = 0.110,
                    iters: int = 180,
                    step: float = 1.0) -> Dict[str, np.ndarray]:
    """简单碰撞分离：把距离小于 (r_i+r_j) 的点沿连线反向各推开一半。"""
    keys = list(keys)
    xy = np.array([pos[k] for k in keys], dtype=float)
    s = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-9)
    radii = base_sep + scale_sep * s
    for _ in range(int(iters)):
        moved = False
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                delta = xy[i] - xy[j]
                dist = float(np.hypot(delta[0], delta[1])) + 1e-9
                min_d = radii[i] + radii[j]
                if dist < min_d:
                    overlap = (min_d - dist) / dist * 0.5 * step
                    shift = delta * overlap
                    xy[i] += shift; xy[j] -= shift
                    moved = True
        step *= 0.96
        if not moved: break
    for k, p in zip(keys, xy):
        pos[k] = p
    return pos

def _compute_importance(G: nx.Graph, metric: str = "strength") -> Dict[str, float]:
    """节点重要性（0~1）：strength/degree/betweenness。"""
    metric = (metric or "strength").lower()
    if metric == "degree":
        vals = dict(G.degree(weight="w_vis"))
    elif metric == "betweenness":
        vals = nx.betweenness_centrality(G, weight="w_vis", normalized=True)
    else:
        vals = {n: 0.0 for n in G.nodes()}
        for u, v in G.edges():
            w = float(G[u][v].get("w_vis", 0.0))
            vals[u] += w; vals[v] += w
    arr = np.array(list(vals.values()), dtype=float)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-12:
        return {n: 0.0 for n in G.nodes()}
    return {n: (vals[n] - mn) / (mx - mn) for n in G.nodes()}

# ====================== 主函数 ======================

def plot_nature_network(
    edges_df: pd.DataFrame,
    nodes_df: Optional[pd.DataFrame] = None,
    *,
    types_path: Optional[Path] = None,
    id2name_path: Optional[Path] = None,
    # 视觉参数
    title: str = "Network",
    label_all: bool = True,
    width_range: Tuple[float,float]=(0.8, 10.0),
    alpha_range: Tuple[float,float]=(0.12, 0.95),
    drop_below: Optional[float] = 0.02,
    node_size_min: float = 360,
    node_size_max: float = 540,
    # 边权处理
    top_edges: Optional[int] = 2000,
    clip_quantile: float = 0.98,
    weight_gamma: float = 2.0,
    # 社区布局与中心化
    keep_gcc: bool = True,
    community_radius: float = 2.0,
    intra_k_scale: float = 6.0,
    radial_exponent: float = 0.92,
    center_metric: str = "strength",
    community_center_pull: float = 0.55,
    community_radius_min: float = 0.6,
    center_pull: float = 0.40,
    topk_pin_to_center: int = 0,
    center_pin_radius: float = 0.25,
    # 防重叠
    collide_base_sep: float = 0.025,
    collide_scale_sep: float = 0.110,
    collide_iters: int = 180,
    # 阴影（比例偏移）
    shadow_frac: float = 0.010,
    shadow_angle_deg: float = 315.0,
    shadow_alpha: float = 0.18,
    # 背景
    bg_color: str = "#FFFFFF",
    # 实体类型图例
    show_types_legend: bool = True,
    # 边按关系类型着色
    color_edges_by_reltype: bool = True,
    rel_colors: Optional[Dict[str, str]] = None,
    show_rel_legend: bool = True,
    # —— 保存相关 —— #
    save_path_base: Optional[Path] = None,   # 例如 Path("out/figure") → 生成 .svg & .pdf
    save_formats: Sequence[str] = ("svg","pdf"),
    save_dpi: int = 600,
) -> None:
    """
    绘制 Nature 风格网络图；若提供 save_path_base，则自动保存为 SVG 与/或 PDF。

    参数
    ----
    edges_df : DataFrame
        包含 u,v（或 source/target 等兼容列）、w_final/置信度等列。
    nodes_df : DataFrame, optional
        至少包含 entity_id；如无类型信息，函数会从 types_path 取主类型。
    save_path_base : Path, optional
        不含扩展名的保存路径前缀；例如 Path("out/fig") → 输出 out/fig.svg 与 out/fig.pdf。
    """

    # 标准化边表
    E = _standardize_edges(edges_df, weight_col="w_final")
    if top_edges is not None and len(E) > top_edges:
        E = E.sort_values("w_norm", ascending=False).head(top_edges)

    # 节点表与类型
    if nodes_df is None or nodes_df.empty or "entity_id" not in nodes_df.columns:
        nd = pd.DataFrame({"entity_id": pd.Index(E["u"]).union(pd.Index(E["v"]))})
    else:
        nd = nodes_df[["entity_id"]].drop_duplicates()
    nd = _attach_node_types(nd, types_path)
    type_map = nd.set_index("entity_id")["node_type"].to_dict()

    # 构图
    G = nx.Graph()
    for u,v,w,rt in E[["u","v","w_norm","reltype_top"]].itertuples(index=False):
        G.add_edge(u, v, w=float(w), reltype=str(rt) if isinstance(rt,str) else "")
    for n in G.nodes():
        G.nodes[n]["node_type"] = type_map.get(n,"Other")

    # 边权可视化映射：削峰 + gamma
    W = np.array([G[u][v]["w"] for u,v in G.edges()], dtype=float)
    if clip_quantile and 0 < clip_quantile < 1 and len(W):
        q = float(np.quantile(W, clip_quantile))
        W = np.clip(W, 0, q) / (q if q>1e-12 else 1.0)
    W = np.power(W, float(weight_gamma))
    for (u,v),wv in zip(G.edges(), W):
        G[u][v]["w_vis"] = float(np.clip(wv,0,1))

    # 保留最大连通分量
    if keep_gcc and G.number_of_nodes()>0:
        comp = max(nx.connected_components(G), key=len)
        G = G.subgraph(comp).copy()

    # 节点重要性（0~1）
    imp = _compute_importance(G, metric=center_metric)

    # 社区优先布局：重要社群靠中心
    try:
        comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight="w_vis"))
    except Exception:
        comms = [set(G.nodes())]

    comm_imp = []
    for c in comms:
        vals = [imp[n] for n in c] if len(c) else [0.0]
        comm_imp.append(float(np.mean(vals)))
    comm_imp = np.array(comm_imp)
    if len(comm_imp):
        comm_imp = (comm_imp - comm_imp.min())/(comm_imp.max()-comm_imp.min()+1e-9)

    # 估算社区间强度
    comm_list = list(comms)
    comm_strength = []
    for i, ci in enumerate(comm_list):
        tot = 0.0
        for j, cj in enumerate(comm_list):
            if i == j: continue
            for u in ci:
                for v in cj:
                    if G.has_edge(u, v):
                        tot += G[u][v].get("w_vis", 0.0)
        comm_strength.append((i, tot))

    order = [i for i, _ in sorted(comm_strength, key=lambda x: x[1], reverse=True)]

    # 用新的顺序放置角度
    centers = {}
    Rmax, Rmin = float(community_radius), float(community_radius_min)
    for k, i in enumerate(order):
        scale = max(Rmin / Rmax, 1.0 - community_center_pull * (comm_imp[i] if len(comm_imp) else 0.0))
        Ri = Rmax * scale
        ang = 2 * np.pi * k / max(len(comm_list), 1)
        centers[i] = np.array([Ri * np.cos(ang), Ri * np.sin(ang)])

    pos = {}
    for i, comm in enumerate(comms):
        sub = G.subgraph(comm)
        k = float(intra_k_scale) / np.sqrt(max(sub.number_of_nodes(), 1))
        sub_pos = nx.spring_layout(sub, weight="w_vis", iterations=80, k=k, seed=42, scale=1.0)
        arr = np.array(list(sub_pos.values()))
        if len(arr):
            arr = (arr - arr.mean(0)) * (1.15 / (np.linalg.norm(arr,axis=1).max()+1e-9))
        for (n,_), p in zip(sub_pos.items(), arr):
            pos[n] = centers[i] + p

    # 全局半径压缩
    if 0 < radial_exponent < 1 and len(pos):
        keys = list(pos.keys())
        arr = np.array([pos[k] for k in keys])
        r = np.linalg.norm(arr, axis=1) + 1e-9
        r_new = np.power(r, float(radial_exponent))
        arr = (arr / r[:,None]) * r_new[:,None]
        for k, p in zip(keys, arr):
            pos[k] = p

    # 节点层中心拉力
    if center_pull > 0 and len(pos):
        for n in list(G.nodes()):
            s = float(np.clip(imp.get(n, 0.0), 0.0, 1.0))
            pos[n] = pos[n] * (1.0 - center_pull * s)

    # Top-K 固定半径
    if topk_pin_to_center and topk_pin_to_center > 0 and len(pos):
        top_nodes = [n for n,_ in sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:int(topk_pin_to_center)]]
        eps = 1e-9
        for n in top_nodes:
            vec = pos[n]
            r = float(np.linalg.norm(vec)) + eps
            pos[n] = vec / r * float(center_pin_radius)

    # 防重叠两轮
    nodes = list(G.nodes())
    sizes = _node_sizes(G, min_size=node_size_min, max_size=node_size_max)
    pos = _repel_overlaps(pos, sizes, nodes, collide_base_sep, collide_scale_sep, collide_iters, step=1.0)
    pos = _repel_overlaps(pos, sizes, nodes,
                      base_sep=collide_base_sep*1.6,
                      scale_sep=collide_scale_sep*1.35,
                      iters=max(80, collide_iters//2), step=0.6)

    # ========= 绘制 =========
    fig, ax = plt.subplots(figsize=(12,12), dpi=200)
    fig.set_facecolor(bg_color); fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color); ax.axis("off")
    wmin,wmax = width_range; amin,amax = alpha_range

    # 边：弧线
    # 取关系颜色映射（优先用户传入）
    REL_COLORS = rel_colors or DEFAULT_REL_COLORS

    # a) 边（弧线）
    for (u, v) in G.edges():
        wv = float(G[u][v]["w_vis"])
        if drop_below is not None and wv < float(drop_below):
            continue
        x1, y1 = pos[u];
        x2, y2 = pos[v]
        lw = wmin + (wmax - wmin) * wv
        alpha = amin + (amax - amin) * wv
        if color_edges_by_reltype:
            rel = G[u][v].get("reltype", "") or "Unknown"
            color = REL_COLORS.get(rel, EDGE_GREY)
            alpha *= REL_ALPHA.get(rel, 0.80)
        else:
            color = EDGE_GREY
        # 伪随机弧度减少边重叠
        s = (hash(u) ^ hash(v)) & 0xFFFF
        rad = ((s % 200) / 200.0 - 0.5) * 0.35
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     connectionstyle=f"arc3,rad={rad}",
                                     arrowstyle="-", linewidth=lw,
                                     color=color, alpha=alpha))

    # 阴影：按比例偏移（避免描边）
    xs = np.array([p[0] for p in pos.values()]); ys = np.array([p[1] for p in pos.values()])
    span_x = xs.max() - xs.min(); span_y = ys.max() - ys.min()
    theta = np.deg2rad(shadow_angle_deg)
    dx = shadow_frac * span_x * np.cos(theta)
    dy = shadow_frac * span_y * np.sin(theta)
    pos_shadow = {n: np.array([pos[n][0] + dx, pos[n][1] + dy]) for n in nodes}
    shadow = nx.draw_networkx_nodes(
        G, pos_shadow, nodelist=nodes, node_size=sizes,
        node_color="#000000", alpha=float(shadow_alpha),
        linewidths=0, edgecolors="none", ax=ax
    )
    if hasattr(shadow, "set_zorder"):
        shadow.set_zorder(1)

    # 节点本体（无描边）
    NODE_TYPE_COLORS = {
        "BMC": "#F9D622","EGR": "#F28D21","ASPKM": "#CC6677","CRD": "#459FC4",
        "APP": "#FF7676","SCN": "#44AA99","AAI": "#117733","CRBC": "#332288",
        "NM":"#AA4499","EF":"#88CCEE","Other":"#CBD5E1",
    }
    cols = [NODE_TYPE_COLORS.get(G.nodes[n]["node_type"], NODE_TYPE_COLORS["Other"]) for n in nodes]
    body = nx.draw_networkx_nodes(
        G, pos, nodelist=nodes, node_size=sizes,
        node_color=cols, edgecolors="none", linewidths=0, ax=ax
    )

    if hasattr(body, "set_zorder"):
        body.set_zorder(3)

    # 标签（白描边可选）
    id2name = _load_id2name(id2name_path)
    if label_all:
        labels = {n: id2name.get(n,n) for n in nodes}
    else:
        deg = dict(G.degree())
        score = {n:(deg.get(n,0)+1)*sizes[i] for i,n in enumerate(nodes)}
        top = [n for n,_ in sorted(score.items(), key=lambda kv:kv[1], reverse=True)[:30]]
        labels = {n:id2name.get(n,n) for n in top}
    texts = nx.draw_networkx_labels(G, pos, labels=labels, font_size=22, ax=ax)
    for t in texts.values():
        t.set_zorder(4)

    # 图例
    # 节点类型图例（已有）
    if show_types_legend:
        uniq_types = sorted({G.nodes[n]["node_type"] for n in nodes})
        handles_nodes = [Patch(color=NODE_TYPE_COLORS.get(t, NODE_TYPE_COLORS["Other"]), label=t) for t in uniq_types]
        if handles_nodes:
            leg_nodes = ax.legend(handles=handles_nodes, title="Node type",
                                  loc="upper right", frameon=True, framealpha=0.95)
            leg_nodes.get_frame().set_facecolor(bg_color)
            ax.add_artist(leg_nodes)  # 关键：保留第一个图例

    # —— 新增：关系类型图例 ——
    if show_rel_legend and color_edges_by_reltype:
        rels_used = sorted({(G[u][v].get("reltype", "") or "Unknown") for u, v in G.edges()})
        handles_rel = [Patch(color=(rel_colors or DEFAULT_REL_COLORS).get(r, EDGE_GREY), label=r)
                       for r in rels_used]
        if handles_rel:
            leg_rel = ax.legend(handles=handles_rel, title="Relation",
                                loc="lower right", frameon=True, framealpha=0.95)
            leg_rel.get_frame().set_facecolor(bg_color)

    ax.set_title(title, fontsize=26, color="#0F172A", pad=8)
    plt.tight_layout()

    # —— 保存为 SVG / PDF —— #
    if save_path_base is not None:
        p = Path(save_path_base)
        p.parent.mkdir(parents=True, exist_ok=True)
        for fmt in save_formats:
            fmt = fmt.lower().strip(".")
            plt.savefig(p.with_suffix(f".{fmt}"),
                        dpi=save_dpi if fmt in ("png","jpg","jpeg") else None,
                        bbox_inches="tight", facecolor=bg_color)

    plt.show()
