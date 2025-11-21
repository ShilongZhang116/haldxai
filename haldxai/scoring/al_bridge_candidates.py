# haldxai/scoring/al_bridge_candidates.py
# -*- coding: utf-8 -*-
"""
Aging / Longevity / Bridge 候选挖掘
-----------------------------------
从已构建的“种子子网/统一子网”中，基于个性化 PageRank（全局）与 1-hop 强度（局部）
计算 A/L 轴向分数与桥接分数，并结合节点质量与结构性度量，产出候选列表。

输入:
- edges.csv: 列包含 ['u','v','w_final', ...]
- nodes.csv: 列包含 ['entity_id','role','degree_sub','strength_sub','name', ...]
- aging_ids.txt, longevity_ids.txt: 每行一个 entity_id

输出(可选落盘):
- <out_dir>/<subnet_tag>__scores_all_nodes.csv
- <out_dir>/<subnet_tag>__Aging_candidates_top{K}.csv
- <out_dir>/<subnet_tag>__Longevity_candidates_top{K}.csv
- <out_dir>/<subnet_tag>__Bridge_candidates_top{K}.csv
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import numpy as np
import pandas as pd
import networkx as nx


# ============== 基础工具 ==============
def normalize01(x: pd.Series | np.ndarray) -> pd.Series:
    """Min-Max 归一，空域或常数列返回 0。"""
    s = pd.to_numeric(pd.Series(x), errors="coerce").fillna(0.0)
    mn, mx = float(s.min()), float(s.max())
    if mx - mn < 1e-12:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index if isinstance(s, pd.Series) else None)
    return (s - mn) / (mx - mn)


def read_id_list(path: Path) -> List[str]:
    """读取换行分隔的 ID 列表，不存在则返回空列表。"""
    if not path or not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_graph(edges: pd.DataFrame, weight_col: str = "w_final") -> nx.Graph:
    """从边表构建无向加权图。"""
    g = nx.Graph()
    for u, v, w in edges[["u", "v", weight_col]].itertuples(index=False):
        g.add_edge(str(u), str(v), w=float(w))
    return g


def personalized_pagerank(G: nx.Graph, seeds: Iterable[str], alpha: float = 0.85) -> pd.Series:
    """个性化 PageRank（若 seeds 为空或不在图中，则返回全 0）。"""
    seeds = [s for s in set(seeds) if s in G]
    if not seeds:
        return pd.Series(0.0, index=list(G.nodes()))
    pers = {n: 0.0 for n in G.nodes()}
    val = 1.0 / len(seeds)
    for s in seeds:
        pers[s] = val
    pr = nx.pagerank(G, alpha=alpha, personalization=pers, weight="w", tol=1e-08, max_iter=200)
    return pd.Series(pr, dtype=float)


def one_hop_weight_to_seeds(G: nx.Graph, seeds: Iterable[str]) -> pd.Series:
    """与种子邻接边权之和（若无 seeds 返回全 0）。"""
    seeds = set(seeds) & set(G.nodes())
    if not seeds:
        return pd.Series(0.0, index=list(G.nodes()))
    acc: Dict[str, float] = {}
    for n in G.nodes():
        s = 0.0
        for nbr, data in G[n].items():
            if nbr in seeds:
                s += float(data.get("w", 0.0))
        acc[n] = s
    return pd.Series(acc, dtype=float)


def top_seed_neighbors(G: nx.Graph, node: str, seeds: Iterable[str], k: int = 6) -> str:
    """列出节点与种子的最强 k 条邻边（用于可解释性）。"""
    seeds = set(seeds)
    if node not in G or not seeds:
        return ""
    tmp: List[Tuple[str, float]] = []
    for nbr, data in G[node].items():
        if nbr in seeds:
            tmp.append((nbr, float(data.get("w", 0.0))))
    tmp.sort(key=lambda x: x[1], reverse=True)
    return ";".join([f"{n}:{w:.3f}" for n, w in tmp[:k]])


def compute_quality_table(weights: pd.DataFrame) -> pd.DataFrame:
    """
    从 entity_type_weights_full 构造质量表：
      - entity_type：优先选择 weight_sum 最大的一条作为主类型；没有则任选一条
      - quality_norm：对 log1p(evidence_count/sources_unique/weight_sum/weight_max/weight_mean/default_hits)
                      的加权和再做 0–1 归一
    只使用存在的列，避免 KeyError。
    """
    df = weights.copy()
    df["entity_id"] = df["entity_id"].astype(str)

    # 主类型行
    if "weight_sum" in df.columns:
        df_main = (df.sort_values(["entity_id", "weight_sum"], ascending=[True, False])
                     .drop_duplicates("entity_id"))
    else:
        df_main = df.drop_duplicates("entity_id")

    # 质量原始分
    comp_weights = {
        "evidence_count": 0.35,
        "sources_unique": 0.25,
        "weight_sum":     0.25,
        "weight_max":     0.10,
        "weight_mean":    0.05,
        "default_hits":   0.10,
    }
    avail = [c for c in comp_weights if c in df_main.columns]
    if not avail:
        df_main["quality_raw"] = 0.0
    else:
        tot = sum(comp_weights[c] for c in avail)
        comp = {c: comp_weights[c] / tot for c in avail}
        qr = 0.0
        for c, w in comp.items():
            x = pd.to_numeric(df_main[c], errors="coerce").fillna(0.0)
            qr = qr + w * np.log1p(x)
        df_main["quality_raw"] = qr

    df_main["quality_norm"] = normalize01(df_main["quality_raw"])
    keep = ["entity_id", "entity_type", "quality_norm"] + \
           [c for c in ["evidence_count","sources_unique","weight_sum","weight_max","weight_mean","default_hits"]
            if c in df_main.columns]
    return df_main[keep].reset_index(drop=True)


# ============== 参数对象 ==============
@dataclass
class CandidateMiningParams:
    """计算与排序的权重参数。"""
    # A/L 轴向综合：PPR 与 1-hop 的比例
    a_ppr_weight: float = 0.6
    a_local_weight: float = 0.4
    l_ppr_weight: float = 0.6
    l_local_weight: float = 0.4

    # 总分线性权重
    w_a_total: Tuple[float, float, float, float] = (0.55, 0.20, 0.15, 0.10)  # A_score, quality, strength, degree
    w_l_total: Tuple[float, float, float, float] = (0.55, 0.20, 0.15, 0.10)  # L_score, quality, strength, degree
    w_b_total: Tuple[float, float, float, float] = (0.60, 0.15, 0.15, 0.10)  # Bridge, strength, degree, quality

    # PageRank 参数
    ppr_alpha: float = 0.85

    # 解释性邻居数量
    k_neighbors: int = 6


# ============== 主流程 ==============
def mine_al_bridge_candidates(
    subnet_dir: Path,
    priors_dir: Path,
    weights_parquet: Path | None,
    out_dir: Path | None = None,
    params: CandidateMiningParams = CandidateMiningParams(),
    topk: int = 200,
) -> dict:
    """
    执行候选挖掘，返回：
    {
      "nodes_all": pd.DataFrame,     # 每个节点的各项分数与总分
      "A_candidates": pd.DataFrame,  # 去种子后的 TopK
      "L_candidates": pd.DataFrame,
      "B_candidates": pd.DataFrame,
      "subnet_tag": str,
      "out_paths": dict | None,      # 若 out_dir 提供则包含落盘路径
    }
    """
    subnet_tag = Path(subnet_dir).name

    # 1) 读边/点
    edges = pd.read_csv(Path(subnet_dir) / "edges_with_confidence.csv")
    nodes = pd.read_csv(Path(subnet_dir) / "nodes.csv")
    edges["u"] = edges["u"].astype(str)
    edges["v"] = edges["v"].astype(str)
    edges["w_final"] = pd.to_numeric(edges["w_final"], errors="coerce").fillna(0.0)

    G = build_graph(edges, weight_col="w_final")

    # 2) 读 seeds
    aging_ids = read_id_list(Path(priors_dir) / "aging_ids.txt")
    longevity_ids = read_id_list(Path(priors_dir) / "longevity_ids.txt")
    S_A = set(aging_ids) & set(G.nodes())
    S_L = set(longevity_ids) & set(G.nodes())

    # 3) 轴向分数
    A_ppr = personalized_pagerank(G, S_A, alpha=params.ppr_alpha)
    L_ppr = personalized_pagerank(G, S_L, alpha=params.ppr_alpha)
    A_loc = one_hop_weight_to_seeds(G, S_A)
    L_loc = one_hop_weight_to_seeds(G, S_L)

    A_score = params.a_ppr_weight * normalize01(A_ppr) + params.a_local_weight * normalize01(A_loc)
    L_score = params.l_ppr_weight * normalize01(L_ppr) + params.l_local_weight * normalize01(L_loc)

    # 4) 桥接分
    A_n, L_n = normalize01(A_score), normalize01(L_score)
    Bridge = 2 * A_n * L_n / (A_n + L_n + 1e-12)

    # 5) 合并到节点表，并加质量与结构项
    nodes["entity_id"] = nodes["entity_id"].astype(str)
    feat = nodes.set_index("entity_id")[["role","degree_sub","strength_sub","name"]].copy()

    if weights_parquet and Path(weights_parquet).exists():
        wei = pd.read_parquet(weights_parquet)
        qual = compute_quality_table(wei).set_index("entity_id")
        feat = feat.join(qual, how="left")
    else:
        feat["entity_type"] = None
        feat["quality_norm"] = 0.0

    feat["A_score"] = A_score.reindex(feat.index).fillna(0.0)
    feat["L_score"] = L_score.reindex(feat.index).fillna(0.0)
    feat["Bridge_score"] = Bridge.reindex(feat.index).fillna(0.0)

    deg_n = normalize01(feat["degree_sub"]) if "degree_sub" in feat.columns else pd.Series(0.0, index=feat.index)
    str_n = normalize01(feat["strength_sub"]) if "strength_sub" in feat.columns else pd.Series(0.0, index=feat.index)
    qual_n = feat["quality_norm"].fillna(0.0)

    a1, a2, a3, a4 = params.w_a_total
    l1, l2, l3, l4 = params.w_l_total
    b1, b2, b3, b4 = params.w_b_total

    feat["A_total"] = a1*feat["A_score"] + a2*qual_n + a3*str_n + a4*deg_n
    feat["L_total"] = l1*feat["L_score"] + l2*qual_n + l3*str_n + l4*deg_n
    feat["B_total"] = b1*feat["Bridge_score"] + b2*str_n + b3*deg_n + b4*qual_n

    seeds_all = S_A | S_L
    feat["is_seed"] = feat.index.isin(seeds_all)

    # 6) 排序与解释性邻居
    def _mk(df: pd.DataFrame, col: str, seeds: set[str]) -> pd.DataFrame:
        out = (df[~df["is_seed"]]
               .sort_values(col, ascending=False)
               .reset_index().rename(columns={"index": "entity_id"}))
        if col.startswith("A_"):
            out["A_top_neighbors"] = out["entity_id"].map(lambda n: top_seed_neighbors(G, n, seeds, k=params.k_neighbors))
        if col.startswith("L_"):
            out["L_top_neighbors"] = out["entity_id"].map(lambda n: top_seed_neighbors(G, n, seeds, k=params.k_neighbors))
        return out

    A_candidates = _mk(feat, "A_total", S_A).head(topk)
    L_candidates = _mk(feat, "L_total", S_L).head(topk)
    B_candidates = (feat[~feat["is_seed"]]
                    .sort_values("B_total", ascending=False)
                    .reset_index().rename(columns={"index": "entity_id"})
                    .head(topk))

    # 7) 统一加 subnet 标签
    nodes_all = feat.reset_index().rename(columns={"index": "entity_id"})
    for df in (nodes_all, A_candidates, L_candidates, B_candidates):
        df.insert(0, "subnet", subnet_tag)

    # 8) 可选落盘
    out_paths = None
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        def _save(df: pd.DataFrame, name: str) -> Path:
            p = out_dir / f"{subnet_tag}__{name}.csv"
            df.to_csv(p, index=False, encoding="utf-8-sig")
            return p
        out_paths = {
            "nodes_all": _save(nodes_all, "scores_all_nodes"),
            "A":        _save(A_candidates, f"Aging_candidates_top{topk}"),
            "L":        _save(L_candidates, f"Longevity_candidates_top{topk}"),
            "B":        _save(B_candidates, f"Bridge_candidates_top{topk}"),
        }

    return {
        "nodes_all": nodes_all,
        "A_candidates": A_candidates,
        "L_candidates": L_candidates,
        "B_candidates": B_candidates,
        "subnet_tag": subnet_tag,
        "out_paths": out_paths,
    }


__all__ = [
    "CandidateMiningParams",
    "mine_al_bridge_candidates",
]
