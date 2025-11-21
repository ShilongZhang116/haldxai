# -*- coding: utf-8 -*-
"""
HALD · relation_types 基于来源的加权聚合（有向/无向）
==================================================
输入字段（至少）：
['rel_pk','relation_id','source_entity_id','target_entity_id',
 'source_entity_name','target_entity_name','relation_type','source']

主要入口
--------
compute_relation_edges(input_path_or_df, ...)
    → (edges_dir, edges_undir, id2name, src_cov, rtype_cov, meta)

说明
----
• 关系类型标准化：仅当命中 REL_LISTS（同义词表）时折叠为 5 类；未命中时保留原始文本。
• 来源权重：同一行命中多个来源时取“最大权重”（可按需改为均值/求和）。
• 无向合并：合并 (a→b)/(b→a)，并保留方向度量（weight_uv/weight_vu, dir_ratio, dir_prefer）。
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Union, Iterable
from collections import Counter, defaultdict
from dataclasses import dataclass
import json, re, time
import numpy as np
import pandas as pd

# --- 可选进度条 ---
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# ---------------- 默认配置（可在调用时覆盖） ----------------

DEFAULT_USECOLS = [
    "source_entity_id", "target_entity_id",
    "source_entity_name", "target_entity_name",
    "relation_type", "source",
]

# 来源权重（可覆盖）
DEFAULT_REL_SOURCE_WT: Dict[str, float] = {
    'ageanno': 1.0, 'agingatlas': 1.0, 'ctd': 1.0, 'hall': 1.0, 'hetionet': 1.0,
    'opengenes': 1.0, 'primekg': 1.0, 'umls': 1.0,
    'JCRQ1-IF10-DeepSeekV3': 0.8, 'AgingRelated-DeepSeekV3': 0.8,
    'JCRQ1-IF10-DeepSeekR1-32B': 0.7, 'AgingRelated-DeepSeekR1-32B': 0.7,
    'JCRQ1-IF10-DeepSeekR1-7B': 0.6, 'AgingRelated-DeepSeekR1-7B': 0.6,
    'bert_model_prediction': 0.4,
}
DEFAULT_REL_SOURCE_ALIAS: Dict[str, str] = {
    "uniprot": "opengenes",   # 示例：把不同写法归并
}
DEFAULT_WT: float = 0.5       # 未命中来源时使用的默认权重

# 关系类型同义词 → 5 类；未命中时保留原始文本
REL_LISTS: Dict[str, List[str]] = {
    "Causal": [
        "Causal","CAusal","CAUSAL","causal","Causes","Causality","Causation",
        "Causal Relationships","Causal Relationship","Causative","Cause","cause",
        "Association & Causality","Association & Causal","Association & Causation",
        "Influence","Influences","Influenced by","Influenced",
        "Affects","Induces","Direct","Inverse","Outcome",
        "Predictive","Mediation","Mediating","Etiological",
        "Evidence Supporting","Evidence-based","Protective","Treatment",
        "Intervention","Involved in","Causative Relationship","Cause-effect",
    ],
    "Association": [
        "Association","ASSOCIATION","Associative","associative","Associational",
        "Associated","ASSOCIATED","Associated with","associated with","Associated With",
        "ASSOC","Association Relationships","Association Relationship",
        "Association & Linkage","Association & Link",
        "Association & Correlation","Association &","Connection","ASSOCIATIVE",
        "No association","Negative association","Negative Association",
        "Positive","Comparison","Compared","Correlation",
        "Observational","Relevant","Evaluation","Trend","Process","Descriptive",
        "Exploratory","No Relationship","associated",
        "Associative Relationships","Relationship","Associative Relationship",
        "Positive association","Positive Association","association",
    ],
    "Regulatory": [
        "Regulatory","REGULATORY","regulatory","Regulate",
        "Regulatory Relationships","Regulatory Relationship",
        "Regulates","REG","Regulation","REGulatory",
    ],
    "Structural/Belonging": [
        "Structural/Belonging","Structural/Belonging Relationships",
        "Structural/Belonging Relationship","structural/belonging",
        "STRUCTURAL/BELONGING","structural/Belonging","Part Relationship",
        "Structural","structural","STRUCTURAL","Structural Relationship",
        "Structural relationship","structural relationship",
        "Belonging","BELONGING","Belonging Relationships",
        "Belongs to","Part-Whole","Part of","Part",
        "component of","Component","Subset","Whole",
        "Instance","Classification","Part-of",
        "APP feature","BELONGS_TO","Structural/Biological",
    ],
    "Interaction & Feedback": [
        "Interaction & Feedback","INTERACTION & FEEDBACK","interaction & feedback",
        "Interaction & Feedback Relationships","Interaction","Feedback",
        "Association & Feedback","Association & Interaction",
        "Competitive","Connected","INTERACTION &FEEDBACK",
    ],
}
_REL_CANON_MAP = {}
for canon, lst in REL_LISTS.items():
    for alias in lst:
        _REL_CANON_MAP[(alias or "").strip().lower()] = canon


def canon_or_raw(s: str) -> str:
    """命中 5 类映射则返回标准名；否则保留原文（去首尾空白）。"""
    raw = (s or "").strip()
    return _REL_CANON_MAP.get(raw.lower(), raw)


# ---------------- 来源权重工具 ----------------

def _build_lookup(source_wt: Dict[str, float]):
    wt_lower = {k.lower(): v for k, v in source_wt.items()}
    keys_lower = sorted(wt_lower.keys(), key=len, reverse=True)
    return keys_lower, wt_lower


def _canon_token(tok: str, alias: Dict[str, str]) -> str:
    """标准化 source token：去空白、小写、`__` 前缀、去括号、别名映射。"""
    if not isinstance(tok, str):
        return ""
    s = tok.strip().lower()
    if not s:
        return ""
    if "__" in s:
        s = s.split("__", 1)[0]
    s = s.replace(" ", "").strip("[]()")
    if s in alias:
        s = alias[s]
    return s


def infer_rel_source_weight(
    source_field: str,
    *,
    source_wt: Dict[str, float] = DEFAULT_REL_SOURCE_WT,
    source_alias: Dict[str, str] = DEFAULT_REL_SOURCE_ALIAS,
    default_wt: float = DEFAULT_WT,
) -> Tuple[float, List[str], bool]:
    """
    将一条 `source` 映射为行级权重。
    返回：(row_weight, 命中来源键列表(小写), 是否用默认权重)
    规则：命中多个来源时取“最大权重”（可按需改为均值/求和）。
    """
    if not isinstance(source_field, str) or not source_field.strip():
        return float(default_wt), [], True

    keys_lower, wt_lower = _build_lookup(source_wt)
    cand = set()
    for t in re.split(r"[;,|]+", source_field):
        ctok = _canon_token(t, source_alias)
        if not ctok:
            continue
        if ctok in wt_lower:         # 等值命中
            cand.add(ctok)
            continue
        for key in keys_lower:       # 宽松包含
            if key in ctok:
                cand.add(key)
                break

    if cand:
        return float(max(wt_lower[c] for c in cand)), sorted(list(cand)), False
    return float(default_wt), [], True


# ---------------- 参数与读取 ----------------

@dataclass
class AggParams:
    """聚合参数。"""
    min_edge_weight: int = 1           # 出现次数阈值（过滤稀疏边）
    chunksize: int = 1_000_000         # CSV 流式分块大小
    sample_frac: float | None = None   # 行级抽样比例（None=不抽样）
    random_state: int = 42             # 抽样随机种子


def _iter_chunks(path: Path, usecols: List[str], chunksize: int) -> Iterable[pd.DataFrame]:
    """
    统一的分块读取器：Parquet 一次读入；CSV 流式按块。
    """
    if path.suffix.lower() == ".parquet":
        yield pd.read_parquet(path, columns=usecols)
    else:
        yield from pd.read_csv(
            path, dtype=str, usecols=lambda c: c in set(usecols),
            chunksize=chunksize, low_memory=False
        )


# ---------------- 主函数 ----------------

def compute_relation_edges(
    input_path_or_df: Union[str, Path, pd.DataFrame],
    *,
    params: AggParams = AggParams(),
    usecols: List[str] = DEFAULT_USECOLS,
    source_wt: Dict[str, float] = DEFAULT_REL_SOURCE_WT,
    source_alias: Dict[str, str] = DEFAULT_REL_SOURCE_ALIAS,
    default_wt: float = DEFAULT_WT,
    keep_reltype_json: bool = False,   # False：仅保留 reltype_top（省内存）
    keep_sources: bool = False,        # False：不统计 sources_unique（更省内存）
    keep_names: bool = False,          # False：不做 id→name 映射（最省内存）
    save_dir: Union[str, Path, None] = None,
) -> Tuple[pd.DataFrame, Dict[str, str], pd.DataFrame, pd.DataFrame, Dict]:
    """
    仅计算【无向】边（低内存版，带进度）：
      - 流式读取；每行直接聚合到 (u=min, v=max)
      - 可选是否保留关系类型分布 / 来源集合 / 名称映射
    返回：
      edges_undir, id2name(dict), src_cov(DataFrame), rtype_cov(DataFrame), meta(dict)
    """
    t0 = time.time()

    # 输入统一：路径或 DataFrame
    if isinstance(input_path_or_df, pd.DataFrame):
        chunks = [input_path_or_df[usecols].copy()]
    else:
        path = Path(input_path_or_df)
        chunks = _iter_chunks(path, usecols, params.chunksize)

    # --- 无向累积容器（键为 (u,v)） ---
    und_cnt  = Counter()              # 次数
    und_wsum = Counter()              # 权重和
    und_wmax = defaultdict(float)     # 最大行权重
    und_def  = Counter()              # 使用默认权重的行数
    und_top  = {}                     # 仅跟踪 reltype 的“胜出者” (label, score)
    und_labs = defaultdict(Counter)   # 若 keep_reltype_json=True 则完整统计分布
    und_srcu = defaultdict(set)       # 若 keep_sources=True 则记录来源集合

    # 覆盖率
    src_hit_ctr = Counter()
    src_def_rows = 0
    rtype_ctr = Counter()

    # 名称映射（可选）
    name_ctr = defaultdict(Counter) if keep_names else None

    def key_uv(a, b):  # 规范键
        return (a, b) if a <= b else (b, a)

    # --- 进度条/日志 ---
    pbar = tqdm(total=None, desc="聚合关系（无向）", mininterval=1.0) if _HAS_TQDM else None
    _last_log_t = time.time()

    # --- 主循环 ---
    for i, chunk in enumerate(chunks, start=1):
        c = chunk.dropna(subset=["source_entity_id", "target_entity_id"]).copy()
        if params.sample_frac and 0 < params.sample_frac < 1:
            c = c.sample(frac=params.sample_frac, random_state=params.random_state + i, replace=False)
            if c.empty:
                # 原来：if pbar:
                if pbar is not None:
                    if (i % 10 == 0) or (time.time() - _last_log_t > 3):
                        pbar.set_postfix({
                            "chunks": i,
                            "pairs": f"{len(und_cnt):,}",
                            "edges": f"{sum(und_cnt.values()):,}",
                        })
                        pbar.update(0)
                        _last_log_t = time.time()
                elif i % 50 == 0:
                    print(f"[{i:>6} chunks] pairs={len(und_cnt):,}  edges={sum(und_cnt.values()):,}")
                continue

        a = c["source_entity_id"].astype(str).values
        b = c["target_entity_id"].astype(str).values
        rlab = c["relation_type"].astype(str).fillna("").map(canon_or_raw).values

        res = c["source"].map(lambda s: infer_rel_source_weight(
            s, source_wt=source_wt, source_alias=source_alias, default_wt=default_wt
        ))
        row_w = res.map(lambda x: x[0]).astype(float).values
        hit_s = res.map(lambda x: ";".join(x[1]) if x[1] else "").values
        used_default = res.map(lambda x: x[2]).values

        if keep_names:
            sname = c.get("source_entity_name", pd.Series([""]*len(c))).fillna("").astype(str).values
            tname = c.get("target_entity_name", pd.Series([""]*len(c))).fillna("").astype(str).values

        for idx in range(len(a)):
            aa = a[idx]; bb = b[idx]; w = row_w[idx]; hs = hit_s[idx]
            isdef = used_default[idx]; rt = rlab[idx]
            if not aa or not bb or aa == bb:
                continue
            u, v = key_uv(aa, bb)

            und_cnt[(u, v)]  += 1
            und_wsum[(u, v)] += float(w)
            if w > und_wmax[(u, v)]:
                und_wmax[(u, v)] = float(w)
            if isdef:
                und_def[(u, v)] += 1

            # 关系类型统计：极省内存模式只跟踪“胜出者”
            if keep_reltype_json:
                und_labs[(u, v)][rt] += 1
            else:
                if (u, v) not in und_top:
                    und_top[(u, v)] = (rt, 1)
                else:
                    cur_lab, cur_cnt = und_top[(u, v)]
                    und_top[(u, v)] = (rt, cur_cnt + 1) if rt == cur_lab else (cur_lab, cur_cnt - 1 if cur_cnt > 1 else 1)

            # 来源覆盖率
            if hs:
                for k in hs.split(";"):
                    if k:
                        src_hit_ctr[k] += 1
                        if keep_sources:
                            und_srcu[(u, v)].add(k)
            else:
                src_def_rows += 1

            rtype_ctr[rt] += 1

            # 名称计数
            if keep_names:
                sn = sname[idx]; tn = tname[idx]
                if sn:
                    name_ctr[aa][sn] += 1
                if tn:
                    name_ctr[bb][tn] += 1

        # --- 刷新进度 ---
        # 原来：if pbar:
        if pbar is not None:
            if (i % 10 == 0) or (time.time() - _last_log_t > 3):
                pbar.set_postfix({
                    "chunks": i,
                    "pairs": f"{len(und_cnt):,}",
                    "edges": f"{sum(und_cnt.values()):,}",
                })
                pbar.update(0)
                _last_log_t = time.time()
        elif i % 50 == 0:
            print(f"[{i:>6} chunks] pairs={len(und_cnt):,}  edges={sum(und_cnt.values()):,}")

    # 原来：if pbar:
    if pbar is not None:
        pbar.close()

    # 组装无向 DataFrame（并做阈值过滤）
    rows = []
    for (u, v), cnt in und_cnt.items():
        if cnt < params.min_edge_weight:
            continue
        wsum = float(und_wsum[(u, v)])
        wmax = float(und_wmax[(u, v)])
        defhits = int(und_def[(u, v)])
        if keep_reltype_json:
            tctr = und_labs[(u, v)]
            top_rt = tctr.most_common(1)[0][0] if tctr else ""
            rel_json = json.dumps(dict(tctr.most_common(6)), ensure_ascii=False)
        else:
            top_rt = und_top.get((u, v), ("", 0))[0]
            rel_json = None

        rows.append({
            "u": u, "v": v,
            "count": int(cnt),
            "weight_sum": wsum,
            "weight_mean": wsum / max(1, cnt),
            "weight_max": wmax,
            "default_hits": defhits,
            "any_default": bool(defhits > 0),
            "sources_unique": (len(und_srcu[(u, v)]) if keep_sources else None),
            "reltype_top": top_rt,
            **({"reltype_json": rel_json} if keep_reltype_json else {}),
        })

    edges_undir = pd.DataFrame(rows).sort_values(["weight_sum","count"], ascending=False).reset_index(drop=True)

    id2name = {}
    if keep_names and name_ctr:
        id2name = {eid: cnt.most_common(1)[0][0] for eid, cnt in name_ctr.items() if len(cnt) > 0}

    src_cov = (pd.DataFrame({"hit_key": list(src_hit_ctr.keys()), "rows": list(src_hit_ctr.values())})
               .sort_values("rows", ascending=False).reset_index(drop=True))
    _, wt_lower = _build_lookup(source_wt)
    src_cov["mapped_weight"] = src_cov["hit_key"].map(wt_lower)
    src_cov.loc[len(src_cov)] = {"hit_key": "__DEFAULT__", "rows": src_def_rows, "mapped_weight": default_wt}

    rtype_cov = (pd.DataFrame({"reltype": list(rtype_ctr.keys()), "rows": list(rtype_ctr.values())})
                 .sort_values("rows", ascending=False).reset_index(drop=True))

    meta = {
        "mode": "undir_lowmem",
        "min_edge_weight": params.min_edge_weight,
        "chunksize": params.chunksize,
        "sample_frac": params.sample_frac,
        "keep_reltype_json": keep_reltype_json,
        "keep_sources": keep_sources,
        "keep_names": keep_names,
        "undirected_edges": int(len(edges_undir)),
        "runtime_sec": round(time.time() - t0, 2),
    }

    if save_dir is not None:
        out = Path(save_dir); out.mkdir(parents=True, exist_ok=True)
        edges_undir.to_parquet(out / "edges_undir_full.parquet", index=False)
        src_cov.to_csv(out / "rel_source_coverage.csv", index=False, encoding="utf-8-sig")
        rtype_cov.to_csv(out / "reltype_coverage.csv", index=False, encoding="utf-8-sig")
        (out / "id2name.json").write_text(json.dumps(id2name, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return edges_undir, id2name, src_cov, rtype_cov, meta
