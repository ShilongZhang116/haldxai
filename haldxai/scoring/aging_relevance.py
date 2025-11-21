# -*- coding: utf-8 -*-
"""
HALD · 实体“衰老相关性”打分与排名（通用：BMC / AAI / ...）
================================================================
功能概述
--------
给定一个目标实体类型（例如 BMC / AAI），从“实体证据(entity_evidence)”与“文章信息(articles)”
出发，结合 aging_prob、期刊影响因子、文献新近度等，计算每个实体的 AgingScore，并导出排名表。
支持：
  • 直接传入“种子实体集合”（CSV/DataFrame/列表），或
  • 无种子时自动从 entity_type_weights_full.parquet 兜底筛选该类型的高可信实体

输出（可选落盘）：
  - <etype>_aging_rank.csv            # 评分排名表
  - <etype>_top_entities.json         # 前 Top-N 的实体清单（便于画子网）

输入数据最低要求
----------------
- entity_evidence.csv ：至少包含 ['pmid','entity_id']
- articles.csv        ：至少包含 ['pmid','aging_prob','factor','pub_date','journal','journal_abbr','title']
- entity_type_weights_full.parquet（或 DataFrame）兜底筛种子时要求包含：
    ['entity_id','entity_type','weight_norm','evidence_count','sources_unique','rank_in_entity']

作者：你 :)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Iterable, Tuple, Dict, List

import json
import numpy as np
import pandas as pd


# ---------------- 参数结构 ----------------

@dataclass
class FallbackSelection:
    """
    当 seeds_input 未提供时，用于从“类型聚合权重表”中筛选某类型实体作为种子。
    """
    top_primary_only: bool = True              # 仅取主类型（rank_in_entity == 1）
    min_weight_norm: Optional[float] = 0.60    # 固定阈值；None 表示走分位数
    percentile_q: float = 0.85                 # min_weight_norm=None 时使用该分位数
    min_evidence_count: int = 2
    min_sources_unique: int = 1
    case_insensitive_type: bool = False        # 类型名大小写是否不敏感


@dataclass
class AgingScoreParams:
    """
    文章级加权与打分参数
    """
    aging_threshold: float = 0.70      # aging_prob ≥ 此阈值计入 n_pmids_aging_high
    use_journal_weight: bool = True    # 是否使用期刊因子
    use_recency_weight: bool = True    # 是否使用新近度
    half_life_years: float = 6.0       # 文献新近度半衰期（年）
    alpha: float = 1.0                 # 行打分中的 aging_prob 的幂指数（≥1 强化高相关文章）
    evidence_chunksize: int = 2_000_000  # entity_evidence 的分块大小


# ---------------- 小工具 ----------------

def _norm01(arr: np.ndarray | pd.Series) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx <= mn:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)

def _recency_weight(pub_date: pd.Timestamp, today: pd.Timestamp, half_life_years: float) -> float:
    """权重 = 0.5 ** (年龄(年)/half_life_years)；缺失日期→1.0"""
    if pd.isna(pub_date): return 1.0
    age_days = (today - pub_date).days
    if age_days <= 0: return 1.0
    age_years = age_days / 365.25
    return float(0.5 ** (age_years / max(half_life_years, 1e-9)))

def _load_id2name(id2name_path: Optional[Union[str, Path]]) -> Dict[str, str]:
    if not id2name_path: return {}
    p = Path(id2name_path)
    if not p.exists(): return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _load_seeds(
    *,
    entity_type: str,
    seeds_input: Optional[Union[str, Path, pd.DataFrame, Iterable[str]]],
    weights_input: Optional[Union[str, Path, pd.DataFrame]],
    fallback: FallbackSelection
) -> pd.DataFrame:
    """
    返回 DataFrame[entity_id, (可选)weight_norm,evidence_count,sources_unique,rank_in_entity]
    若 seeds_input 提供：
       - str/Path → 读取 CSV（需有 entity_id 列）
       - DataFrame → 直接使用（需有 entity_id 列）
       - Iterable[str] → 构造一列 entity_id
    否则：从 weights_input（parquet/csv/DataFrame）兜底筛选该类型实体。
    """
    if seeds_input is not None:
        if isinstance(seeds_input, (str, Path)):
            df = pd.read_csv(seeds_input, dtype=str, low_memory=False)
        elif isinstance(seeds_input, pd.DataFrame):
            df = seeds_input.copy()
        else:
            # 迭代器或列表
            df = pd.DataFrame({"entity_id": [str(x) for x in seeds_input]})
        assert "entity_id" in df.columns, "seeds_input 需要包含列 entity_id"
        df["entity_id"] = df["entity_id"].astype(str)
        return df

    # —— 兜底：从 weights_input 中筛某类型的高可信实体 ——
    if weights_input is None:
        raise ValueError("未提供 seeds_input，且缺少 weights_input 兜底筛选。")

    if isinstance(weights_input, pd.DataFrame):
        w = weights_input.copy()
    else:
        p = Path(weights_input)
        if p.suffix.lower() == ".parquet":
            w = pd.read_parquet(p)
        else:
            w = pd.read_csv(p, dtype=str, low_memory=False)

    need = {"entity_id","entity_type","weight_norm","evidence_count","sources_unique","rank_in_entity"}
    miss = need - set(w.columns)
    if miss:
        raise ValueError(f"weights_input 缺列: {miss}")

    w["entity_id"] = w["entity_id"].astype(str)
    w["entity_type"] = w["entity_type"].astype(str)
    for c in ("weight_norm",):
        w[c] = pd.to_numeric(w[c], errors="coerce").fillna(0.0)
    for c in ("evidence_count","sources_unique","rank_in_entity"):
        w[c] = pd.to_numeric(w[c], errors="coerce").fillna(0).astype(int)

    if fallback.case_insensitive_type:
        mask = w["entity_type"].str.lower() == entity_type.lower()
    else:
        mask = w["entity_type"] == entity_type
    sub = w[mask].copy()

    if sub.empty:
        return pd.DataFrame({"entity_id":[]})

    if fallback.top_primary_only and "rank_in_entity" in sub.columns:
        sub = sub[sub["rank_in_entity"] == 1].copy()

    if fallback.min_weight_norm is None:
        thr = float(sub["weight_norm"].quantile(fallback.percentile_q))
    else:
        thr = float(fallback.min_weight_norm)

    keep = (
        (sub["weight_norm"] >= thr) &
        (sub["evidence_count"] >= int(fallback.min_evidence_count)) &
        (sub["sources_unique"] >= int(fallback.min_sources_unique))
    )
    return sub.loc[keep, ["entity_id","weight_norm","evidence_count","sources_unique","rank_in_entity"]].drop_duplicates()


def rank_entities_by_aging(
    *,
    entity_type: str,                                 # 目标类型：如 "BMC" / "AAI"
    seeds_input: Optional[Union[str, Path, pd.DataFrame, Iterable[str]]] = None,
    weights_input: Optional[Union[str, Path, pd.DataFrame]] = None,
    evidence_csv: Union[str, Path],
    articles_csv: Union[str, Path],
    id2name_path: Optional[Union[str, Path]] = None,
    fallback: Optional[FallbackSelection] = None,
    scoring: Optional[AgingScoreParams] = None,
    top_articles_k: int = 3,                          # 每个实体记录前 K 篇代表文章
    out_dir: Optional[Union[str, Path]] = None,
    top_n_export: int = 150                           # 导出前 N 实体清单
) -> Tuple[pd.DataFrame, List[str]]:
    """
    主函数：计算某一“实体类型”的 AgingScore 排名。

    返回：
      df  : DataFrame，包含（至少）以下列
            ['entity_id','name','AgingScore','n_pmids_total','n_pmids_aging_high',
             'aging_prob_mean','aging_prob_sum','score_sum', f'{etype}_weight_norm'(可选), 'top_articles']
      seed_ids_used : 实际使用的实体ID列表（便于复用）
    """
    fallback = fallback or FallbackSelection()
    scoring = scoring or AgingScoreParams()

    # 1) 载入/筛出“种子实体集合”
    seeds_df = _load_seeds(
        entity_type=entity_type,
        seeds_input=seeds_input,
        weights_input=weights_input,
        fallback=fallback
    )
    if seeds_df.empty:
        raise RuntimeError(f"没有可用的 {entity_type} 实体种子。")
    seed_ids: set[str] = set(seeds_df["entity_id"].astype(str))
    # 保存一份权重列名（若存在），用于后续合并
    has_weight_norm = "weight_norm" in seeds_df.columns

    # 2) 文章表
    art_cols_need = {"pmid","aging_prob","factor","pub_date","journal","journal_abbr","title"}
    arts = pd.read_csv(articles_csv, dtype={"pmid":str}, low_memory=False)
    miss = art_cols_need - set(arts.columns)
    if miss:
        raise ValueError(f"articles.csv 缺列: {miss}")

    arts["aging_prob"] = pd.to_numeric(arts["aging_prob"], errors="coerce").clip(0, 1).fillna(0.0)
    arts["factor"] = pd.to_numeric(arts["factor"], errors="coerce").fillna(0.0)
    arts["pub_date"] = pd.to_datetime(arts["pub_date"], errors="coerce")
    arts = arts.dropna(subset=["pmid"]).copy()

    # 期刊 & 新近度权重
    if scoring.use_journal_weight:
        jf = np.log1p(arts["factor"].to_numpy())
        jf01 = _norm01(jf)
        arts["w_journal"] = 1.0 + jf01     # 压到 [1,2]
    else:
        arts["w_journal"] = 1.0

    if scoring.use_recency_weight:
        today = pd.Timestamp.today()
        arts["w_recency"] = arts["pub_date"].map(lambda d: _recency_weight(d, today, scoring.half_life_years))
    else:
        arts["w_recency"] = 1.0

    arts["w_article_base"] = arts["w_journal"] * arts["w_recency"]
    art_map = arts.set_index("pmid")[["aging_prob","w_article_base","title","journal","journal_abbr","factor"]]

    # 3) 证据表（分块）
    ev_usecols = {"pmid","entity_id"}
    pairs = []  # 收集 (entity_id, pmid)
    for i, ch in enumerate(pd.read_csv(evidence_csv, dtype=str, low_memory=False,
                                       usecols=lambda c: c in ev_usecols,
                                       chunksize=scoring.evidence_chunksize), start=1):
        ch = ch.dropna(subset=["pmid","entity_id"]).copy()
        ch["entity_id"] = ch["entity_id"].astype(str)
        ch = ch[ch["entity_id"].isin(seed_ids)]
        if ch.empty:
            continue
        ch = ch[["entity_id","pmid"]].drop_duplicates()
        pairs.append(ch)
        # 可选打印：每 10 块提示一次
        # if i % 10 == 0:
        #     print(f"[evidence] chunk {i:>3} 累计对数: {sum(len(x) for x in pairs):,}")

    pairs_df = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame(columns=["entity_id","pmid"])
    if pairs_df.empty:
        raise RuntimeError("过滤后没有 (entity_id, pmid) 与 articles 匹配，请检查数据与路径。")

    # 过滤掉 articles 里没有的 pmid
    pairs_df = pairs_df[pairs_df["pmid"].isin(art_map.index)].copy()
    if pairs_df.empty:
        raise RuntimeError("匹配到的 PMID 在 articles 中不存在。")

    # 连接文章指标 + 行打分
    pairs_df = pairs_df.join(art_map, on="pmid", how="left")
    pairs_df["aging_prob"] = pd.to_numeric(pairs_df["aging_prob"], errors="coerce").fillna(0.0)
    pairs_df["w_article_base"] = pd.to_numeric(pairs_df["w_article_base"], errors="coerce").fillna(1.0)
    pairs_df["row_score"] = (pairs_df["aging_prob"].astype(float) ** float(scoring.alpha)) * pairs_df["w_article_base"].astype(float)

    # 4) 聚合到实体级
    AGING_T = float(scoring.aging_threshold)
    agg = pairs_df.groupby("entity_id", as_index=False).agg(
        n_pmids_total=("pmid","nunique"),
        aging_prob_mean=("aging_prob","mean"),
        aging_prob_sum=("aging_prob","sum"),
        score_sum=("row_score","sum"),
        n_pmids_aging_high=("aging_prob", lambda s: int((s >= AGING_T).sum())),
    )

    # AgingScore = 0-1 归一化(score_sum)
    agg["AgingScore"] = _norm01(agg["score_sum"].to_numpy())

    # —— 每个实体 Top-K 文章（不使用 groupby.apply，避免未来警告）——
    topk_rows = (pairs_df.sort_values(["entity_id","row_score"], ascending=[True, False])
                        .groupby("entity_id", sort=False, group_keys=False)
                        .head(int(top_articles_k))
                        [["entity_id","pmid","row_score","aging_prob","journal_abbr","factor","title"]])

    def _fmt_rows(df: pd.DataFrame) -> List[Dict]:
        out = []
        for _, r in df.iterrows():
            out.append({
                "pmid": str(r.pmid),
                "score": float(r.row_score) if pd.notna(r.row_score) else 0.0,
                "aging_prob": float(r.aging_prob) if pd.notna(r.aging_prob) else 0.0,
                "journal": (r.journal_abbr or ""),
                "factor": float(r.factor) if pd.notna(r.factor) else 0.0,
                "title": (r.title or "")[:120],
            })
        return out

    topk_map: Dict[str, List[Dict]] = {eid: _fmt_rows(gdf) for eid, gdf in topk_rows.groupby("entity_id", sort=False)}
    agg["top_articles"] = agg["entity_id"].map(topk_map)

    # 合并名字与（若有）原始类型权重
    id2name = _load_id2name(id2name_path)
    agg["name"] = agg["entity_id"].map(lambda x: id2name.get(str(x), ""))

    if has_weight_norm:
        agg = agg.merge(seeds_df[["entity_id","weight_norm"]], on="entity_id", how="left")
        agg.rename(columns={"weight_norm": f"{entity_type}_weight_norm"}, inplace=True)

    # 排序
    agg = agg.sort_values(["AgingScore","n_pmids_total","aging_prob_mean"], ascending=[False, False, False]).reset_index(drop=True)

    # 落盘
    seed_list_sorted = sorted(seed_ids)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        etag = entity_type.lower().replace("/","_").replace("\\","_")

        cols = [c for c in [
            "entity_id","name","AgingScore","n_pmids_total","n_pmids_aging_high",
            "aging_prob_mean","aging_prob_sum","score_sum", f"{entity_type}_weight_norm","top_articles"
        ] if c in agg.columns]
        (out / f"{etag}_aging_rank.csv").write_text(
            agg[cols].to_csv(index=False, encoding="utf-8-sig"), encoding="utf-8"
        )
        (out / f"{etag}_top_entities.json").write_text(
            json.dumps(agg.head(int(top_n_export))["entity_id"].astype(str).tolist(), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    return agg, seed_list_sorted
