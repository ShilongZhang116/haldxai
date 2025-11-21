# -*- coding: utf-8 -*-
"""
HALD · 关系网络重要子图抽取（缓存到 cache/aggregation/network）
===============================================================
功能概览
--------
基于 relation_types（三元组归并表）构建“实体—实体”无向网络，
并抽取“最重要节点构成的子图”，将数据落盘以便后续可视化/分析。

核心流程
  1) 流式分块聚合无向边 (u, v)：
       u = min(source_entity_id, target_entity_id)
       v = max(source_entity_id, target_entity_id)
       weight = 出现次数（计数）
       可选：统计每条边的 relation_type 分布，生成 reltype_top / reltype_json
  2) 从边表反推节点统计：
       - degree   ：邻居数（无权）
       - strength ：加权度（∑ incident edge weight）
  3) 节点评分（归一化后线性加权）并选择 Top-N
  4) 导出产物：
       - edges_all_agg.parquet   （全量聚合边）
       - nodes_all_agg.csv       （全量节点统计）
       - nodes_top.csv / edges_top.csv
       - graph_top.gexf          （Gephi / Cytoscape 可直接打开）
       - meta.json               （运行参数与摘要）
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter, defaultdict
import json
import time

import numpy as np
import pandas as pd
import networkx as nx

# 可选：tqdm 进度（未安装不影响）
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


# ======================== 参数定义 ========================

@dataclass
class PipelineParams:
    """控制整个流水线的参数。"""
    # 读入
    chunksize: int = 400_000                 # CSV 分块大小
    usecols: Optional[List[str]] = None      # 自定义列；None 用默认列
    # 边聚合
    min_edge_weight: int = 2                 # 聚合后最小边权阈值
    compute_type_dist: bool = True           # 是否统计 relation_type 分布
    # 节点评分
    alpha_deg: float = 0.6                   # score 中 degree 的权重
    alpha_str: float = 0.4                   # score 中 strength 的权重
    # 子图裁剪
    top_nodes: int = 200                     # 选取前 N 个最重要节点
    top_edges: Optional[int] = 5000          # 子图中保留的最大边数（降序），None 不限


# 默认读取列（relation_types 至少应包含这些列）
_DEFAULT_USECOLS = [
    "source_entity_id", "target_entity_id",
    "source_entity_name", "target_entity_name",
    "relation_type",
]


# ======================== 边聚合（矢量化） ========================

def aggregate_edges_stream(
    path: Union[str, Path],
    *,
    min_edge_weight: int = 1,
    chunksize: int = 400_000,
    usecols: Optional[List[str]] = None,
    compute_type_dist: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    将 relation_types 以“流式分块 + 矢量化 groupby”聚合为无向边 (u,v,weight)。
    可选：统计每条边的 relation_type 分布并给出 top / json（前 5）。

    返回：
      edges_agg: DataFrame['u','v','weight',('reltype_top','reltype_json')]
      name_map:  {entity_id -> 最常出现的名称}
    """
    t0 = time.time()
    path = Path(path)
    usecols = list(usecols or _DEFAULT_USECOLS)

    # 统计容器（外排式累加）
    parts_pairs: List[pd.DataFrame] = []         # 每块 (u,v,count)
    parts_type:  List[pd.DataFrame] = []         # 每块 (u,v,reltype,count)
    name_ctr = defaultdict(Counter)              # id -> Counter(name)

    # 选择读取方式
    if path.suffix.lower() == ".parquet":
        it = [pd.read_parquet(path, columns=[c for c in usecols if c in _DEFAULT_USECOLS])]
    else:
        it = pd.read_csv(
            path, dtype=str,
            usecols=lambda c: c in set(usecols),
            chunksize=chunksize, low_memory=False
        )

    pbar = tqdm(total=None, desc="聚合无向边", mininterval=1.0) if _HAS_TQDM else None
    last_t = time.time()

    for i, chunk in enumerate(it, start=1):
        c = chunk.dropna(subset=["source_entity_id", "target_entity_id"]).copy()
        if c.empty:
            continue

        # 名称计数（按唯一 (id, name) 计一次，降低频次噪声）
        for id_col, nm_col in [("source_entity_id","source_entity_name"),
                               ("target_entity_id","target_entity_name")]:
            if (id_col in c.columns) and (nm_col in c.columns):
                sub = c[[id_col, nm_col]].dropna()
                if not sub.empty:
                    sub = sub.astype(str)
                    gnm = sub.groupby([id_col, nm_col], sort=False).size().reset_index(name="cnt")
                    for eid, nm, cnt in gnm.itertuples(index=False):
                        if nm:
                            name_ctr[eid][nm] += int(cnt)

        # 生成规范化无向边 (u, v)
        a = c["source_entity_id"].astype(str)
        b = c["target_entity_id"].astype(str)
        u = a.where(a <= b, b)
        v = b.where(a <= b, a)
        # 去自环
        mask = u.ne(v)
        u, v = u[mask], v[mask]

        # —— (u,v) 聚合计数
        pairs = pd.DataFrame({"u": u, "v": v})
        g = pairs.groupby(["u","v"], sort=False).size().reset_index(name="weight")
        parts_pairs.append(g)

        # —— (u,v,relation_type) 计数（可选）
        if compute_type_dist and ("relation_type" in c.columns):
            rlab = c.loc[mask, "relation_type"].astype(str).fillna("")
            tri = pd.DataFrame({"u": u, "v": v, "reltype": rlab})
            gt = tri.groupby(["u","v","reltype"], sort=False).size().reset_index(name="cnt")
            parts_type.append(gt)

        # 进度
        if pbar is not None and ((i % 10 == 0) or (time.time() - last_t > 2.0)):
            pbar.set_postfix({"chunks": i, "pairs_rows": f"{sum(x.shape[0] for x in parts_pairs):,}"})
            pbar.update(0); last_t = time.time()
        elif pbar is None and i % 50 == 0:
            print(f"[{i:>6} chunks] pairs_rows={sum(x.shape[0] for x in parts_pairs):,}")

    if pbar is not None:
        pbar.close()

    # —— 全量外排式归并
    if parts_pairs:
        edges_agg = pd.concat(parts_pairs, ignore_index=True)
        edges_agg = (edges_agg.groupby(["u","v"], as_index=False, sort=False)
                             .agg(weight=("weight","sum")))
        edges_agg = edges_agg[edges_agg["weight"] >= int(min_edge_weight)].reset_index(drop=True)
    else:
        edges_agg = pd.DataFrame(columns=["u","v","weight"])

    if compute_type_dist and parts_type:
        tri_all = pd.concat(parts_type, ignore_index=True)
        tri_all = (tri_all.groupby(["u","v","reltype"], as_index=False, sort=False)
                          .agg(cnt=("cnt","sum")))
        # 求每个 (u,v) 的 top reltype
        idx = tri_all.groupby(["u","v"])["cnt"].idxmax()
        top = tri_all.loc[idx, ["u","v","reltype"]].rename(columns={"reltype": "reltype_top"})
        # top5 json
        top5 = (tri_all.sort_values(["u","v","cnt"], ascending=[True, True, False])
                        .groupby(["u","v"])
                        .apply(lambda df: json.dumps(dict(df.head(5)[["reltype","cnt"]].values), ensure_ascii=False))
                        .reset_index(name="reltype_json"))
        edges_agg = edges_agg.merge(top, on=["u","v"], how="left") \
                             .merge(top5, on=["u","v"], how="left")
    else:
        edges_agg["reltype_top"] = ""
        edges_agg["reltype_json"] = ""

    # 名称映射：出现频次最高的名字
    name_map = {eid: cnt.most_common(1)[0][0] for eid, cnt in name_ctr.items() if len(cnt) > 0}

    print(f"[aggregate] edges={len(edges_agg):,} (min_w={min_edge_weight}), "
          f"nodes(seen)={len(name_map):,}, time={time.time()-t0:.1f}s")
    return edges_agg.sort_values("weight", ascending=False).reset_index(drop=True), name_map


# ======================== 节点统计/打分/裁剪 ========================

def build_node_stats_from_edges(edges_agg: pd.DataFrame) -> pd.DataFrame:
    """
    由聚合后的边表反推节点统计（degree/strength）。
    矢量化实现：不使用 iterrows，适合大图。
    约定：edges_agg 至少包含 ['u','v','weight']，且 (u,v) 唯一表示一条无向边。
    """
    if edges_agg.empty:
        return pd.DataFrame(columns=["entity_id","degree","strength"])

    # degree：一个节点的“不同邻居数”。无向简单图下等于 incident 边数（因 (u,v) 唯一）
    deg = pd.concat([edges_agg["u"], edges_agg["v"]], ignore_index=True).value_counts()

    # strength：与该节点相连的边的 weight 之和（两端各加一次）
    str_u = edges_agg.groupby("u")["weight"].sum()
    str_v = edges_agg.groupby("v")["weight"].sum()
    strength = str_u.add(str_v, fill_value=0)

    idx = deg.index.union(strength.index)
    nodes = pd.DataFrame({"entity_id": idx})
    nodes["degree"] = nodes["entity_id"].map(deg).fillna(0).astype(int)
    nodes["strength"] = nodes["entity_id"].map(strength).fillna(0).astype(int)
    return nodes.sort_values(["degree", "strength"], ascending=[False, False]).reset_index(drop=True)


def score_nodes(nodes_df: pd.DataFrame, *, alpha_deg: float = 0.6, alpha_str: float = 0.4) -> pd.DataFrame:
    """score = α·norm(degree) + (1-α)·norm(strength)。"""
    df = nodes_df.copy()
    for col in ["degree", "strength"]:
        m, M = df[col].min(), df[col].max()
        df[f"{col}_norm"] = (df[col] - m) / (M - m) if M > m else 0.0
    df["score"] = alpha_deg * df["degree_norm"] + (1 - alpha_deg) * df["strength_norm"]
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def extract_top_subgraph(
    edges_agg: pd.DataFrame,
    nodes_scored: pd.DataFrame,
    *,
    top_nodes: int = 200,
    top_edges: Optional[int] = 5000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    根据打分选 Top-N 节点，返回：
      nodes_top（含 degree_sub/strength_sub）与 edges_top（按 weight 降序）
    """
    keep = set(nodes_scored.head(top_nodes)["entity_id"])
    sub_edges = edges_agg[edges_agg.u.isin(keep) & edges_agg.v.isin(keep)].copy()
    sub_edges = sub_edges.sort_values("weight", ascending=False)
    if top_edges is not None:
        sub_edges = sub_edges.head(top_edges)

    nodes_top = nodes_scored[nodes_scored["entity_id"].isin(keep)].copy()
    # 子图内的度/强度（便于绘图时按子图统计）
    deg_ctr = defaultdict(int)
    str_ctr = Counter()
    for u, v, w in sub_edges[["u","v","weight"]].itertuples(index=False):
        deg_ctr[u] += 1; deg_ctr[v] += 1
        str_ctr[u] += int(w); str_ctr[v] += int(w)
    nodes_top["degree_sub"] = nodes_top["entity_id"].map(deg_ctr).fillna(0).astype(int)
    nodes_top["strength_sub"] = nodes_top["entity_id"].map(str_ctr).fillna(0).astype(int)
    return nodes_top.reset_index(drop=True), sub_edges.reset_index(drop=True)


# ======================== 落盘工具 ========================

def save_artifacts(
    out_dir: Union[str, Path],
    *,
    edges_all: pd.DataFrame,
    nodes_all: pd.DataFrame,
    nodes_top: pd.DataFrame,
    edges_top: pd.DataFrame,
    name_map: Dict[str, str],
    meta: Dict
) -> None:
    """
    将所有产物保存到 out_dir：
      - edges_all_agg.parquet / nodes_all_agg.csv
      - nodes_top.csv / edges_top.csv
      - graph_top.gexf
      - meta.json
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # 名称合并
    for df in (nodes_all, nodes_top):
        df["name"] = df["entity_id"].map(name_map).fillna("")

    # 全量
    edges_all.to_parquet(out / "edges_all_agg.parquet", index=False)
    nodes_all.to_csv(out / "nodes_all_agg.csv", index=False, encoding="utf-8-sig")

    # 子图
    nodes_top.to_csv(out / "nodes_top.csv", index=False, encoding="utf-8-sig")
    edges_top.to_csv(out / "edges_top.csv", index=False, encoding="utf-8-sig")

    # GEXF（Gephi/Cytoscape）
    G = nx.Graph()
    for r in edges_top.itertuples(index=False):
        attrs = {"weight": int(getattr(r, "weight", 1))}
        if "reltype_top" in edges_top.columns:
            attrs["reltype_top"] = getattr(r, "reltype_top", "")
        G.add_edge(r.u, r.v, **attrs)
    for r in nodes_top.itertuples(index=False):
        n = getattr(r, "entity_id")
        G.nodes[n]["label"] = getattr(r, "name", "")
        G.nodes[n]["degree_sub"] = int(getattr(r, "degree_sub", 0))
        G.nodes[n]["strength_sub"] = int(getattr(r, "strength_sub", 0))
        G.nodes[n]["degree_all"] = int(getattr(r, "degree", 0))
        G.nodes[n]["strength_all"] = int(getattr(r, "strength", 0))
    nx.write_gexf(G, out / "graph_top.gexf")

    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] wrote to: {out.resolve()}")


# ======================== 一键式入口 ========================

def cache_important_subgraph(
    relation_types_path: Union[str, Path],
    out_dir: Union[str, Path],
    params: PipelineParams = PipelineParams(),
    *,
    return_data: bool = True
):
    """
    一键执行：边聚合 → 节点统计 → 打分 → 子图裁剪 → 落盘。

    返回（return_data=True）：
      edges_all, nodes_all, nodes_top, edges_top, name_map, meta
    """
    t0 = time.time()

    edges_agg, name_map = aggregate_edges_stream(
        relation_types_path,
        min_edge_weight=params.min_edge_weight,
        chunksize=params.chunksize,
        usecols=params.usecols or _DEFAULT_USECOLS,
        compute_type_dist=params.compute_type_dist,
    )

    nodes_all = build_node_stats_from_edges(edges_agg)
    nodes_scored = score_nodes(nodes_all, alpha_deg=params.alpha_deg, alpha_str=params.alpha_str)
    nodes_top, edges_top = extract_top_subgraph(
        edges_agg, nodes_scored, top_nodes=params.top_nodes, top_edges=params.top_edges
    )

    meta = {
        "relation_types_path": str(Path(relation_types_path).resolve()),
        "min_edge_weight": params.min_edge_weight,
        "chunksize": params.chunksize,
        "compute_type_dist": params.compute_type_dist,
        "alpha_deg": params.alpha_deg,
        "alpha_str": params.alpha_str,
        "top_nodes": params.top_nodes,
        "top_edges": params.top_edges,
        "edges_all": int(len(edges_agg)),
        "nodes_all": int(len(nodes_all)),
        "edges_top": int(len(edges_top)),
        "nodes_top": int(len(nodes_top)),
        "runtime_sec": round(time.time() - t0, 2),
    }

    save_artifacts(
        out_dir,
        edges_all=edges_agg,
        nodes_all=nodes_all,
        nodes_top=nodes_top,
        edges_top=edges_top,
        name_map=name_map,
        meta=meta,
    )

    if return_data:
        return edges_agg, nodes_all, nodes_top, edges_top, name_map, meta
    return None
