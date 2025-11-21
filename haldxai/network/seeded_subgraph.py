# -*- coding: utf-8 -*-
"""
HALDxAI · 基于种子的通用子网构建器（含实体/边清洗与先验）
================================================================
功能概述
--------
1) 从 relation_types（CSV/Parquet/已加载 DataFrame）中，
   以“种子实体”为核心，分块流式选出每个种子的 Top-K 一阶邻居，
   再用“种子∪邻居”的节点集合回扫全表聚合无向边并计算置信度。

2) 可选：结合实体类型聚合产物（entity_type_weights_full.parquet）构建
   “高质量实体池”（按类型白名单、主类型/权重/证据阈值、名称黑名单），
   用于过滤掉 man / woman / yes / patients 等低生物学含义节点。

3) 可选：对边进行语义约束与类型对先验加权：
   - 关系类型白名单（如 Causal/Regulatory/Interaction & Feedback）
   - 类型对先验 type-pair prior（如 BMC–EGR=1.0、BMC–APP=0.8 等）
   - 分位数/Top-K 截断，得到更干净、可视化友好的子网。

输入最小需求
------------
relation_types 至少包含列：
  ["source_entity_id","target_entity_id","relation_type","source"]

可选输入
--------
- entity_type_weights_full.parquet（或 DataFrame）：用于“高质量实体池”
- id2name.json：节点名称映射，用于标签与名称黑名单过滤（非必需）

输出
----
返回 (edges_df, nodes_df)，并可选落盘 CSV：
  edges_df: u, v, count, weight_sum, weight_mean, weight_max,
            reltype_top, confidence, conf_norm_mm, conf_norm_q, w_final(可选)
  nodes_df: entity_id, role(Seed/Neighbor), degree_sub, strength_sub, name(可选)

作者
----
HALDxAI
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Set, Iterable, Optional, Union
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import json, re

# ======================= 默认参数/映射 =======================

DEFAULT_REL_SOURCE_WT: Dict[str, float] = {
    "ageanno": 1.0, "agingatlas": 1.0, "ctd": 1.0, "hall": 1.0, "hetionet": 1.0,
    "opengenes": 1.0, "primekg": 1.0, "umls": 1.0,
    "JCRQ1-IF10-DeepSeekV3": 0.9, "AgingRelated-DeepSeekV3": 0.9,
    "JCRQ1-IF10-DeepSeekR1-32B": 0.8, "AgingRelated-DeepSeekR1-32B": 0.8,
    "JCRQ1-IF10-DeepSeekR1-7B": 0.7, "AgingRelated-DeepSeekR1-7B": 0.7,
    "bert_model_prediction": 0.6,
}
DEFAULT_REL_SOURCE_ALIAS: Dict[str, str] = {
    "uniprot": "opengenes",
}

REL_LISTS: Dict[str, List[str]] = {
    "Causal": [
        "Causal","CAUSAL","causal","Causes","Causality","Causation","Causal Relationships",
        "Causal Relationship","Causative","Cause","cause","Association & Causality","Association & Causal",
        "Association & Causation","Influence","Influences","Influenced by","Influenced","Affects","Induces",
        "Direct","Inverse","Outcome","Predictive","Mediation","Mediating","Etiological",
        "Evidence Supporting","Evidence-based","Protective","Treatment","Intervention","Involved in",
        "Causative Relationship","Cause-effect",
    ],
    "Association": [
        "Association","ASSOCIATION","Associative","associative","Associational","Associated","ASSOCIATED",
        "Associated with","associated with","Associated With","ASSOC","Association Relationships",
        "Association Relationship","Association & Linkage","Association & Link","Association & Correlation",
        "Association &","Connection","ASSOCIATIVE","No association","Negative association","Negative Association",
        "Positive","Comparison","Compared","Correlation","Observational","Relevant","Evaluation","Trend",
        "Process","Descriptive","Exploratory","No Relationship","associated","Associative Relationships",
        "Relationship","Associative Relationship","Positive association","Positive Association","association",
    ],
    "Regulatory": [
        "Regulatory","REGULATORY","regulatory","Regulate","Regulatory Relationships","Regulatory Relationship",
        "Regulates","REG","Regulation","REGulatory",
    ],
    "Structural/Belonging": [
        "Structural/Belonging","Structural/Belonging Relationships","Structural/Belonging Relationship",
        "structural/belonging","STRUCTURAL/BELONGING","structural/Belonging","Part Relationship","Structural",
        "structural","STRUCTURAL","Structural Relationship","Structural relationship","structural relationship",
        "Belonging","BELONGING","Belonging Relationships","Belongs to","Part-Whole","Part of","Part",
        "component of","Component","Subset","Whole","Instance","Classification","Part-of","APP feature",
        "BELONGS_TO","Structural/Biological",
    ],
    "Interaction & Feedback": [
        "Interaction & Feedback","INTERACTION & FEEDBACK","interaction & feedback",
        "Interaction & Feedback Relationships","Interaction","Feedback",
        "Association & Feedback","Association & Interaction","Competitive","Connected","INTERACTION &FEEDBACK",
    ],
}
REL_CANON = {(a or "").strip().lower(): canon for canon, lst in REL_LISTS.items() for a in lst}

# 名称黑名单（过滤低语义节点）
NAME_BLACKLIST = {
    "patient","patients","subject","subjects","participant","participants",
    "man","men","woman","women","male","female","males","females",
    "yes","no","cohort","group","trial","study","studies","control","controls",
    "baseline","outcome","outcomes","risk","risks","hazard","dose","dosing",
    "placebo","visit","year","years","month","months","day","days", "up",
    "No - yes/no indicator", "Tau - Statistical Technique", 'standard error', 'people'
}

ENTITY_ID_BLACKLIST = {
    "Entity-63e110fccc", 'Entity-d0ea838d90', 'Entity-7fe077596a', 'Entity-735f274bef', 'Entity-5396f3bcea', 'Entity-2242ebcb0b', 'Entity-9ce000ac7d', 'Entity-e6e3a9fd7c',
    'Entity-19c4a593f9', 'Entity-cd1d69a1f0', 'Entity-d0e2ec0918', 'Entity-cc039ef6a9', 'Entity-7d4d3186ce', 'Entity-4f6bd4edc5', 'Entity-5415eb2af2', 'Entity-0c403a1560',
    'Entity-dfb25a2b44', 'Entity-6848ae6f8e', 'Entity-1a68ad6efa', 'Entity-eaf141d01b'
}

# 类型对先验（可扩展；对称处理）
DEFAULT_TYPE_PAIR_PRIOR: Dict[Tuple[str, str], float] = {
    ("BMC","EGR"): 1.0, ("BMC","AAI"): 1.0, ("BMC","ASPKM"): 0.95,
    ("BMC","APP"): 0.80, ("BMC","CRD"): 0.85, ("BMC","NM"): 0.80, ("BMC","SCN"): 0.75,
}

DEFAULT_ALLOWED_RELTYPES = {"Causal","Regulatory","Interaction & Feedback", "Association"}

HALD_CLASSES = [
    "BMC","EGR","ASPKM","CRD","APP","SCN","AAI","CRBC","NM","EF"
]

# ======================= 数据类参数 =======================

@dataclass
class SeededParams:
    # 关系来源权重
    source_wt: Dict[str, float] = field(default_factory=lambda: DEFAULT_REL_SOURCE_WT.copy())
    source_alias: Dict[str, str] = field(default_factory=lambda: DEFAULT_REL_SOURCE_ALIAS.copy())
    default_row_wt: float = 0.4

    # 选邻居规模
    top_k_per_seed: int = 25
    max_neighbors_total: int = 300

    # 第二遍回扫参数
    chunk_size: int = 2_000_000
    min_edge_count: int = 1
    conf_quantile: float = 0.99

    # 实体池（可选）阈值
    use_entity_pool: bool = True
    allowed_types: Iterable[str] = field(default_factory=lambda: tuple(HALD_CLASSES))
    only_primary_type: bool = True
    min_weight_norm: float = 0.60
    min_evidence_count: int = 2
    min_sources_unique: int = 2
    case_insensitive_type: bool = False

    # 名称黑名单
    filter_bad_names: bool = True

    # 实体ID黑名单
    filter_bad_entity_id: bool = True

    # 边清洗/先验
    use_reltype_whitelist: bool = True
    reltype_whitelist: Set[str] = field(default_factory=lambda: set(DEFAULT_ALLOWED_RELTYPES))
    type_pair_prior: Dict[Tuple[str,str], float] = field(default_factory=lambda: DEFAULT_TYPE_PAIR_PRIOR.copy())
    keep_top_edges: Optional[int] = None         # None 表示不截断
    min_w_final_quantile: float = 0.20           # 去掉最弱的 20%



# ======================= 工具函数 =======================

# ---- 名称清洗：去掉 "wt allele" / "wild-type allele" 等 ----
WT_ALLELE_PAT = re.compile(r"\b(?:wt|wild[-\s]?type)\s*allele\b", flags=re.IGNORECASE)

def _clean_entity_name(name: str) -> str:
    """
    去除名称中的 'wt allele' / 'wild-type allele'，并做简单收尾清洗。
    示例: 'KDM1A wt Allele' -> 'KDM1A'；'CD274 Wild-Type Allele' -> 'CD274'
    """
    if not isinstance(name, str):
        return ""
    s = WT_ALLELE_PAT.sub("", name)
    s = re.sub(r"\s{2,}", " ", s)              # 多空格→单空格
    s = s.strip(" \t\r\n,;:()[]{}")            # 去两端标点/空白
    return s


def _canon_token(tok: str, alias_lower: Dict[str,str]) -> str:
    if not isinstance(tok, str): return ""
    s = tok.strip().lower()
    if not s: return ""
    if "__" in s: s = s.split("__", 1)[0]
    s = s.replace(" ", "").strip("[]()")
    if s in alias_lower: s = alias_lower[s]
    return s

def infer_rel_source_weight(source_field: str,
                            source_wt: Dict[str,float],
                            source_alias: Dict[str,str],
                            default_wt: float) -> float:
    """解析 source 字段，命中多个来源时取最大权重；未命中用默认权重"""
    if not isinstance(source_field, str) or not source_field.strip():
        return float(default_wt)
    wt_lower = {k.lower(): v for k, v in source_wt.items()}
    alias_lower = {k.lower(): v for k, v in source_alias.items()}
    keys_sorted = sorted(wt_lower.keys(), key=len, reverse=True)

    best = float(default_wt)
    for t in re.split(r"[;,|]+", source_field):
        ct = _canon_token(t, alias_lower)
        if not ct:
            continue
        if ct in wt_lower:
            best = max(best, float(wt_lower[ct])); continue
        for key in keys_sorted:
            if key in ct:
                best = max(best, float(wt_lower[key])); break
    return best

def canon_or_raw_reltype(s: str) -> str:
    raw = (s or "").strip()
    key = raw.lower()
    return REL_CANON.get(key, raw)

def _iter_relation_chunks(rel: Union[str, Path, pd.DataFrame],
                          usecols: List[str],
                          chunksize: int):
    """支持 CSV/Parquet/DataFrame 的统一迭代器"""
    if isinstance(rel, pd.DataFrame):
        yield rel[usecols].copy(); return
    p = Path(rel)
    if p.suffix.lower() == ".parquet":
        yield pd.read_parquet(p, columns=usecols)
    else:
        for ch in pd.read_csv(p, dtype=str, usecols=lambda c: c in set(usecols),
                              chunksize=chunksize, low_memory=False):
            yield ch

def _bad_name(name: str) -> bool:
    if not isinstance(name, str): return True
    s = name.strip().lower()
    if not s: return True
    if s in NAME_BLACKLIST or s.rstrip('s') in NAME_BLACKLIST:
        return True
    if len(s) < 2 or len(s) > 60: return True
    if re.fullmatch(r"[\W_]+", s): return True
    if s in {"na","n/a","none","unknown"}: return True
    return False

def _bad_entity_id(entity_id: str) -> bool:
    if entity_id in ENTITY_ID_BLACKLIST:
        return True
    return False

def _load_id2name(path: Optional[Union[str, Path]]) -> Dict[str,str]:
    if not path: return {}
    p = Path(path)
    if not p.exists(): return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        # 统一转成 str，并做 wt allele 清洗
        clean = {}
        for k, v in raw.items():
            ks = str(k)
            vs = _clean_entity_name(str(v)) if v is not None else ""
            clean[ks] = vs
        return clean
    except Exception:
        return {}

def _normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

# ======================= 实体池：高质量实体 =======================

def build_high_quality_entity_pool(
    weights_input: Union[str, Path, pd.DataFrame],
    *,
    allowed_types: Iterable[str],
    only_primary: bool,
    min_weight_norm: float,
    min_evidence: int,
    min_sources: int,
    id2name: Optional[Dict[str,str]] = None,
    case_insensitive_type: bool = False,
    filter_bad_names: bool = True,
    filter_bad_entity_id: bool = True
) -> pd.DataFrame:
    """返回实体池 DataFrame[entity_id, entity_type, (name?)]"""
    if isinstance(weights_input, pd.DataFrame):
        df = weights_input.copy()
    else:
        p = Path(weights_input)
        df = pd.read_parquet(p) if p.suffix.lower()==".parquet" else pd.read_csv(p, dtype=str)
    need = {"entity_id","entity_type","weight_norm","evidence_count","sources_unique","rank_in_entity"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"entity weights 缺少必要列: {miss}")

    df["entity_id"] = df["entity_id"].astype(str)
    df["entity_type"] = df["entity_type"].astype(str)
    # 类型过滤
    if case_insensitive_type:
        allow = {t.lower() for t in allowed_types}
        df = df[df["entity_type"].str.lower().isin(allow)]
    else:
        df = df[df["entity_type"].isin(allowed_types)]
    # 主类型
    if only_primary and "rank_in_entity" in df.columns:
        df = df[df["rank_in_entity"].astype("Int64") == 1]
    # 数值阈值
    for c in ("weight_norm","evidence_count","sources_unique"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    keep = (df["weight_norm"] >= float(min_weight_norm)) & \
           (df["evidence_count"] >= int(min_evidence)) & \
           (df["sources_unique"] >= int(min_sources))
    df = df[keep].copy().drop_duplicates(subset=["entity_id"])

    # 名称过滤
    if id2name:
        df["name"] = df["entity_id"].map(lambda x: id2name.get(str(x), ""))
        df["name"] = df["name"].map(_clean_entity_name)
        if filter_bad_names:
            df = df[~df["name"].map(_bad_name)].copy()
        if filter_bad_entity_id:
            df = df[~df["entity_id"].map(_bad_entity_id)].copy()
    return df[["entity_id","entity_type"] + (["name"] if "name" in df.columns else [])].reset_index(drop=True)

# ======================= 主流程 1/2：选邻居 + 回扫 =======================

def find_top_neighbors_per_seed(
    seeds: Set[str],
    relation_types: Union[str, Path, pd.DataFrame],
    *,
    params: SeededParams
) -> pd.DataFrame:
    """
    返回 DataFrame[seed_id, nei_id, count, weight_sum, weight_mean, weight_max, reltype_top]
    仅统计“一端为种子”的行。
    """
    pair_cnt   = Counter()
    pair_wsum  = Counter()
    pair_wmax  = defaultdict(float)
    pair_rtype = defaultdict(Counter)

    usecols = ["source_entity_id","target_entity_id","relation_type","source"]
    for i, ch in enumerate(_iter_relation_chunks(relation_types, usecols, params.chunk_size), start=1):
        c = ch.dropna(subset=["source_entity_id","target_entity_id"]).copy()
        a  = c["source_entity_id"].astype(str).values
        b  = c["target_entity_id"].astype(str).values
        ss = c.get("source", pd.Series([""]*len(c))).values
        rt = c.get("relation_type", pd.Series([""]*len(c))).astype(str).fillna("").map(canon_or_raw_reltype).values
        for aa, bb, srow, rlab in zip(a, b, ss, rt):
            if aa == bb:
                continue
            if aa in seeds and bb not in seeds:
                seed, nei = aa, bb
            elif bb in seeds and aa not in seeds:
                seed, nei = bb, aa
            elif aa in seeds and bb in seeds:
                # 种子间边可计数为“参考”，但不用于扩邻逻辑
                seed, nei = aa, bb
            else:
                continue
            w = infer_rel_source_weight(srow, params.source_wt, params.source_alias, params.default_row_wt)
            key = (seed, nei)  # 种子固定为第一位即可
            pair_cnt[key]  += 1
            pair_wsum[key] += w
            pair_wmax[key]  = max(pair_wmax.get(key, 0.0), w)
            pair_rtype[key][rlab] += 1
        if i % 10 == 0:
            print(f"[pass1] chunk {i}  当前 pair 数: {len(pair_cnt):,}")

    rows = []
    for (seed, nei), cnt in pair_cnt.items():
        tctr = pair_rtype[(seed, nei)]
        top_rt = tctr.most_common(1)[0][0] if tctr else ""
        rows.append({
            "seed_id": seed, "nei_id": nei,
            "count": int(cnt),
            "weight_sum": float(pair_wsum[(seed, nei)]),
            "weight_mean": float(pair_wsum[(seed, nei)] / max(1, cnt)),
            "weight_max": float(pair_wmax[(seed, nei)]),
            "reltype_top": top_rt,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["seed_id","weight_sum","count"], ascending=[True, False, False]).reset_index(drop=True)

def aggregate_subgraph_on_nodes(
    node_ids: Set[str],
    relation_types: Union[str, Path, pd.DataFrame],
    *,
    params: SeededParams
) -> Tuple[pd.DataFrame, Dict[str, Counter]]:
    """
    在 node_ids 上回扫，聚合无向边 + 置信度（confidence/归一化）。
    返回 (edges_df, node_stat)，其中 node_stat 包含 deg/strength（子网内）。
    """
    und_cnt   = Counter()
    und_wsum  = Counter()
    und_wmax  = defaultdict(float)
    und_type  = defaultdict(Counter)

    usecols = ["source_entity_id","target_entity_id","relation_type","source"]
    for i, ch in enumerate(_iter_relation_chunks(relation_types, usecols, params.chunk_size), start=1):
        c = ch.dropna(subset=["source_entity_id","target_entity_id"]).copy()
        mask = c["source_entity_id"].isin(node_ids) & c["target_entity_id"].isin(node_ids)
        c = c[mask]
        if c.empty:
            continue
        a  = c["source_entity_id"].astype(str).values
        b  = c["target_entity_id"].astype(str).values
        ss = c.get("source", pd.Series([""]*len(c))).values
        rt = c.get("relation_type", pd.Series([""]*len(c))).astype(str).fillna("").map(canon_or_raw_reltype).values
        for aa, bb, srow, rlab in zip(a, b, ss, rt):
            if aa == bb:
                continue
            u, v = (aa, bb) if aa <= bb else (bb, aa)
            w = infer_rel_source_weight(srow, params.source_wt, params.source_alias, params.default_row_wt)
            und_cnt[(u,v)]  += 1
            und_wsum[(u,v)] += w
            und_wmax[(u,v)]  = max(und_wmax.get((u,v), 0.0), w)
            und_type[(u,v)][rlab] += 1
        if i % 10 == 0:
            print(f"[pass2] chunk {i}  当前边数: {len(und_cnt):,}")

    rows = []
    for (u,v), cnt in und_cnt.items():
        if cnt < params.min_edge_count:
            continue
        tctr = und_type[(u,v)]
        top_rt = tctr.most_common(1)[0][0] if tctr else ""
        rows.append({
            "u": u, "v": v,
            "count": int(cnt),
            "weight_sum": float(und_wsum[(u,v)]),
            "weight_mean": float(und_wsum[(u,v)] / max(1, cnt)),
            "weight_max": float(und_wmax[(u,v)]),
            "reltype_top": top_rt,
        })
    edges = pd.DataFrame(rows)
    if edges.empty:
        return edges, {"deg": Counter(), "strength": Counter()}

    # confidence & 归一化
    edges["confidence"] = edges["weight_sum"].astype(float)
    c = edges["confidence"].to_numpy()
    cmin, cmax = float(np.min(c)), float(np.max(c))
    edges["conf_norm_mm"] = 0.0 if cmax <= cmin else (edges["confidence"] - cmin) / (cmax - cmin)
    q_hi = float(np.quantile(c, params.conf_quantile))
    c_cap = np.clip(c, 0.0, q_hi)
    cmin2, cmax2 = float(np.min(c_cap)), float(np.max(c_cap))
    edges["conf_norm_q"] = 0.0 if cmax2 <= cmin2 else (np.clip(edges["confidence"], 0.0, q_hi) - cmin2) / (cmax2 - cmin2)

    # 子网内度/强度
    deg = Counter(); strength = Counter()
    for _, r in edges.iterrows():
        u, v = r.u, r.v
        w = float(r.confidence)
        deg[u] += 1; deg[v] += 1
        strength[u] += w; strength[v] += w
    node_stat = {"deg": deg, "strength": strength}

    edges = edges.sort_values(["confidence","count"], ascending=False).reset_index(drop=True)
    return edges, node_stat

# ======================= 主流程 3：边语义与先验 =======================

def _type_pair_prior(a: str, b: str, prior_map: Dict[Tuple[str,str], float]) -> float:
    if a == b:
        return 0.7
    if (a,b) in prior_map:
        return float(prior_map[(a,b)])
    if (b,a) in prior_map:
        return float(prior_map[(b,a)])
    return 0.6

def apply_edge_semantics_and_priors(
    edges_df: pd.DataFrame,
    nodes_types: pd.DataFrame,
    *,
    params: SeededParams
) -> pd.DataFrame:
    """
    根据关系语义白名单 + 类型对先验，得到最终边强度 w_final 并筛选。
    nodes_types 需包含列：entity_id, entity_type
    """
    E = edges_df.copy()
    for c in ("u","v"):
        E[c] = E[c].astype(str)
    # 语义白名单
    if params.use_reltype_whitelist and "reltype_top" in E.columns:
        E["reltype_top"] = E["reltype_top"].fillna("").astype(str)
        E = E[E["reltype_top"].isin(params.reltype_whitelist) | (E["reltype_top"]=="")].copy()

    # 基础强度（优先 conf_norm_q → confidence → weight/count）
    if "conf_norm_q" in E.columns:
        base = pd.to_numeric(E["conf_norm_q"], errors="coerce").fillna(0.0).to_numpy()
    elif "confidence" in E.columns:
        base = pd.to_numeric(E["confidence"], errors="coerce").fillna(0.0).to_numpy()
        base = _normalize01(base)
    else:
        tmp = None
        for k in ("weight","count"):
            if k in E.columns:
                tmp = pd.to_numeric(E[k], errors="coerce").fillna(0.0).to_numpy(); break
        base = _normalize01(tmp if tmp is not None else np.ones(len(E), dtype=float))
    E["w_base"] = base

    # 合并实体类型与类型对先验
    typemap = nodes_types.set_index("entity_id")["entity_type"].to_dict() if not nodes_types.empty else {}
    E["type_u"] = E["u"].map(typemap).fillna("Other")
    E["type_v"] = E["v"].map(typemap).fillna("Other")
    E["prior"]  = [ _type_pair_prior(a,b, params.type_pair_prior) for a,b in zip(E["type_u"], E["type_v"]) ]
    E["w_final"] = E["w_base"] * E["prior"]

    # 分位数阈
    thr = float(E["w_final"].quantile(params.min_w_final_quantile)) if len(E) else 0.0
    E = E[E["w_final"] >= thr].copy()

    # 可选：只保留最强 Top-K 边
    if params.keep_top_edges and params.keep_top_edges > 0:
        E = E.sort_values("w_final", ascending=False).head(params.keep_top_edges)

    # 无向规范 + 去重
    uu = E[["u","v"]].min(axis=1).astype(str); vv = E[["u","v"]].max(axis=1).astype(str)
    E["u"], E["v"] = uu, vv
    agg = (E.groupby(["u","v"], as_index=False)
             .agg(w_final=("w_final","max"),
                  reltype_top=("reltype_top", lambda s: s.value_counts().idxmax() if len(s.dropna()) else "")))
    return agg

# ======================= 统一入口 =======================

def run_seeded_subgraph(
    seeds: Iterable[str],
    relation_types: Union[str, Path, pd.DataFrame],
    *,
    params: SeededParams,
    entity_weights: Optional[Union[str, Path, pd.DataFrame]] = None,
    id2name_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    统一入口：返回 (edges_df, nodes_df)。若 params.output_dir 指定，则写出 CSV。
    """
    seeds = set(map(str, seeds))
    if not seeds:
        raise ValueError("seeds 为空。")

    # 1) 选邻居
    seed_nei = find_top_neighbors_per_seed(seeds, relation_types, params=params)
    if seed_nei.empty:
        raise RuntimeError("没有找到任何一阶邻居，请检查 seeds 或 relation_types。")

    # 2) 每个种子 Top-K + 全局截断
    topk = (seed_nei.sort_values(["seed_id","weight_sum","count"], ascending=[True, False, False])
                  .groupby("seed_id", group_keys=False)
                  .head(params.top_k_per_seed))
    nei_all = list(dict.fromkeys(topk["nei_id"].tolist()))
    if len(nei_all) > params.max_neighbors_total:
        tmp = (topk.groupby("nei_id")["weight_sum"].sum()
                    .sort_values(ascending=False)
                    .head(params.max_neighbors_total))
        nei_all = tmp.index.tolist()

    # 3) 实体池过滤（可选）——尽量在“扩邻结果”阶段过滤
    id2name = _load_id2name(id2name_path)
    pool = pd.DataFrame(columns=["entity_id","entity_type"])
    if params.use_entity_pool and entity_weights is not None:
        pool = build_high_quality_entity_pool(
            entity_weights,
            allowed_types=params.allowed_types,
            only_primary=params.only_primary_type,
            min_weight_norm=params.min_weight_norm,
            min_evidence=params.min_evidence_count,
            min_sources=params.min_sources_unique,
            id2name=id2name,
            case_insensitive_type=params.case_insensitive_type,
            filter_bad_names=params.filter_bad_names,
            filter_bad_entity_id=params.filter_bad_entity_id,
        )
        pool_ids = set(pool["entity_id"].astype(str))
        # 不删种子，只过滤邻居
        nei_all = [n for n in nei_all if n in pool_ids]

    final_nodes: Set[str] = set(seeds) | set(nei_all)

    # 4) 回扫聚合子网边
    edges_sub, node_stat = aggregate_subgraph_on_nodes(final_nodes, relation_types, params=params)

    # 5) 节点表（角色/度/强度/名称）
    nodes = pd.DataFrame({"entity_id": list(final_nodes)})
    nodes["role"] = np.where(nodes["entity_id"].isin(seeds), "Seed", "Neighbor")
    nodes["degree_sub"]   = nodes["entity_id"].map(lambda x: node_stat["deg"].get(x, 0)).astype(int)
    nodes["strength_sub"] = nodes["entity_id"].map(lambda x: node_stat["strength"].get(x, 0.0)).astype(float)
    if id2name:
        nodes["name"] = nodes["entity_id"].map(lambda x: id2name.get(x, ""))
        nodes["name"] = nodes["name"].map(_clean_entity_name)

    # 6) 边清洗 + 先验（若有实体池，则拿 pool 类型；否则节点类型=Other）
    if not pool.empty:
        nodes_types = pool[["entity_id","entity_type"]].copy()
    else:
        nodes_types = nodes[["entity_id"]].copy()
        nodes_types["entity_type"] = "Other"

    edges_final = apply_edge_semantics_and_priors(edges_sub, nodes_types, params=params)

    # 7) 落盘（可选）
    if output_dir:
        out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        edges_sub.to_csv(out_dir / "edges_with_confidence_raw.csv", index=False, encoding="utf-8-sig")
        edges_final.to_csv(out_dir / "edges_with_confidence.csv", index=False, encoding="utf-8-sig")
        nodes.sort_values(["role","degree_sub","strength_sub"], ascending=[True, False, False]) \
             .to_csv(out_dir / "nodes.csv", index=False, encoding="utf-8-sig")

    return edges_final, nodes
