# haldxai/selection/seed_priors.py
# -*- coding: utf-8 -*-
"""
Seed2Subgraph · 种子先验映射（名称→entity_id）
================================================
说明：
- 本模块将“衰老/长寿”先验名称（基因/通路/干预等）映射到 HALD 实体 ID，
  产出固定目录下的一组可复用文件（不使用任何日期标签）。
- 可作为 S2S（Seed2Subgraph）流程的上游步骤，在 Notebook 中直接调用。

输出（固定目录，覆盖式写入）
--------------------------------
<proj_root>/data/HALD-Seed2Subgraph/priors/
  ├─ seed_master.csv
  ├─ seed_map_detail.csv
  ├─ seed_map_summary.csv
  ├─ name_to_ids.json
  ├─ name_multiples.json
  ├─ name_nohits.json
  ├─ aging_ids.txt
  ├─ longevity_ids.txt
  ├─ combined_ids.txt
  └─ README.md

用法见本文件底部示例或 Notebook 片段。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import json, os, re
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


# =========================
# 1) 配置对象（可序列化）
# =========================
@dataclass
class SeedPriorConfig:
    """
    参数：
    - name2id_csv: 必需，至少两列 `name, entity_id`
    - weights_parquet: 可选，含列 `entity_id, entity_type, weight_norm, rank_in_entity`
      * 用于类型过滤、仅保留主类型、按权重排序；若为空则跳过这些步骤
    - allowed_types: 允许的实体类型集合；为空/None 表示不过滤类型（推荐留空=全部允许）
    - min_weight_norm: >0 时按最小权重阈值过滤
    - only_primary: True → 仅保留 `rank_in_entity == 1` 的主类型
    - id_blacklist: 需要排除的 entity_id 集合
    """
    name2id_csv: Path
    weights_parquet: Path | None = None
    allowed_types: Tuple[str, ...] = tuple()     # 留空=不过滤类型
    min_weight_norm: float = 0.0
    only_primary: bool = True
    id_blacklist: Tuple[str, ...] = tuple()


# =========================
# 2) 文本清洗与同义词规则
# =========================
# —— 正则触发词 → 备选正则列表（可按需扩充）——
SYNONYM_REGEX: Mapping[str, Sequence[str]] = {
    r"\bPGC[-\s]?1A\b":      [r"\bPPARGC1A\b", r"\bPGC1A\b", r"PGC-?1α"],
    r"\bAMPK\b":             [r"\bPRKAA1\b", r"\bPRKAA2\b", r"\bAMPK( pathway| signaling)?\b"],
    r"\bFOXO3\b":            [r"\bFOXO-?3\b"],
    r"\bSIRT1\b":            [r"\bSirtuin-?1\b"],
    r"\bMTOR\b":             [r"\bmTOR\b", r"\bMTORC1\b", r"\bMTORC2\b", r"mTOR signaling"],
    r"\bRAPAMYCIN\b":        [r"\bSirolimus\b"],
    r"\bCALORIC RESTRICTION\b": [r"\bCR\b", r"\bCalorie Restriction\b",
                                 r"\bDietary Restriction\b", r"\bDR\b"],
}

_COMPILED_SYNONYM = {
    re.compile(src, re.IGNORECASE): [re.compile(p, re.IGNORECASE) for p in pats]
    for src, pats in SYNONYM_REGEX.items()
}


def clean_name_for_match(name: str) -> str:
    """基础清洗：去尾缀、统一破折号、希腊字母转写、压缩空格。"""
    if not isinstance(name, str):
        return ""
    s = name.strip()
    s = re.sub(r"\b(wt|wild[\-\s]*type)\s*allele\b", "", s, flags=re.IGNORECASE)
    s = s.replace("α", "alpha").replace("β", "beta")
    s = re.sub(r"[\u2010-\u2015]", "-", s)  # 各种破折号统一
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _to_key(s: str) -> str:
    """严格等价匹配键：清洗 + 大写 + 去空格/下划线。"""
    s = clean_name_for_match(s).upper()
    s = re.sub(r"[\s_]+", "", s)
    return s


def _patterns_for_name(name: str) -> List[re.Pattern]:
    """
    给定原始名称，返回“包含式匹配”的正则列表：
    1) 名称本体（清洗后）→ 词边界匹配
    2) 命中同义词触发器 → 追加其候选正则
    """
    pats: List[re.Pattern] = []
    base = clean_name_for_match(name)
    if base:
        if base.isascii() and not any(ch in base for ch in r".*+?^$[](){}|\\" ):
            pats.append(re.compile(rf"\b{re.escape(base.upper())}\b"))
        else:
            pats.append(re.compile(base.upper(), re.IGNORECASE))
    bu = base.upper()
    for src_pat, syn_list in _COMPILED_SYNONYM.items():
        if src_pat.search(bu):
            pats.extend(syn_list)
    return pats


# =========================
# 3) 主流程函数
# =========================
def map_names_to_ids(
    seed_df: pd.DataFrame,
    config: SeedPriorConfig,
) -> tuple[Dict[str, List[str]], pd.DataFrame, pd.DataFrame]:
    """
    将名称映射为 entity_id 列表，并返回：
      name_to_ids: dict[name -> [entity_id,...]]
      detail_df:   每条命中一行（name, mode, entity_id, entity_type, weight_norm, ...）
      summary_df:  每个名称一行（n_ids, ids_preview）
    `seed_df` 至少包含列：name（建议再带 seed_set / category / tier）
    """
    # —— 载入 name2id —— #
    name2id = (
        pd.read_csv(config.name2id_csv, dtype=str)
          .dropna(subset=["name", "entity_id"])
          .copy()
    )
    name2id["name_clean"] = name2id["name"].map(clean_name_for_match)
    name2id["key"] = name2id["name_clean"].map(_to_key)

    # key → {entity_id}
    key2ids: Dict[str, set[str]] = defaultdict(set)
    for _, r in name2id.iterrows():
        key2ids[r["key"]].add(str(r["entity_id"]))

    # 候选池（包含式匹配）
    cand = name2id[["name_clean", "entity_id"]].drop_duplicates().copy()
    cand["UP"] = cand["name_clean"].str.upper()
    cand["entity_id"] = cand["entity_id"].astype(str)

    id_black = set(config.id_blacklist or [])

    # —— 阶段1：严格等价匹配 —— #
    hits: List[dict] = []
    missed: List[str] = []
    for _, row in seed_df.iterrows():
        raw = str(row["name"])
        k = _to_key(raw)
        if k in key2ids:
            for eid in key2ids[k]:
                if eid in id_black:
                    continue
                hits.append({
                    "name": raw, "mode": "exact", "entity_id": eid,
                    "seed_set": row.get("seed_set", ""),
                    "category": row.get("category", ""),
                    "tier": row.get("tier", 1),
                })
        else:
            missed.append(raw)

    # —— 阶段2：包含/同义词正则 —— #
    for raw in missed:
        pats = _patterns_for_name(raw)
        if not pats:
            continue
        mask = pd.Series(False, index=cand.index)
        for pp in pats:
            mask = mask | cand["UP"].map(lambda s: bool(pp.search(s)))
        sub = cand[mask]
        for _, r in sub.iterrows():
            eid = str(r["entity_id"])
            if eid in id_black:
                continue
            hits.append({
                "name": raw, "mode": "regex_contains", "entity_id": eid,
                "seed_set": "", "category": "", "tier": 1,
            })

    detail_df = pd.DataFrame(hits).drop_duplicates()

    # —— 与权重表对齐（可选） —— #
    if config.weights_parquet and Path(config.weights_parquet).exists() and not detail_df.empty:
        wei = pd.read_parquet(
            config.weights_parquet,
            columns=["entity_id", "entity_type", "weight_norm", "rank_in_entity"]
        )
        wei["entity_id"] = wei["entity_id"].astype(str)
        wei["weight_norm"] = pd.to_numeric(wei["weight_norm"], errors="coerce").fillna(0.0)

        if config.only_primary and "rank_in_entity" in wei.columns:
            mask_primary = pd.to_numeric(wei["rank_in_entity"], errors="coerce").fillna(0) == 1
            wei = wei[mask_primary].copy()

        if config.allowed_types:
            wei = wei[wei["entity_type"].isin(set(config.allowed_types))].copy()

        if config.min_weight_norm > 0:
            wei = wei[wei["weight_norm"] >= float(config.min_weight_norm)].copy()

        detail_df = (
            detail_df.merge(wei[["entity_id", "entity_type", "weight_norm"]],
                            on="entity_id", how="left")
                     .sort_values(["name", "mode", "weight_norm"],
                                  ascending=[True, True, False])
                     .drop_duplicates(subset=["name", "entity_id"])
        )
    else:
        if not detail_df.empty:
            detail_df["entity_type"] = None
            detail_df["weight_norm"] = np.nan

    # —— 汇总 —— #
    name_to_ids: Dict[str, List[str]] = defaultdict(list)
    for _, r in detail_df.iterrows():
        name_to_ids[r["name"]].append(str(r["entity_id"]))

    for raw in seed_df["name"].astype(str).unique():
        name_to_ids.setdefault(raw, [])

    # 去重保序
    name_to_ids = {k: list(dict.fromkeys(v)) for k, v in name_to_ids.items()}

    summary_rows = []
    for raw in seed_df["name"].astype(str).unique():
        ids = name_to_ids[raw]
        summary_rows.append({
            "name": raw, "n_ids": len(ids), "ids_preview": ";".join(ids[:12])
        })
    summary_df = (
        pd.DataFrame(summary_rows)
          .sort_values(["n_ids", "name"], ascending=[False, True])
          .reset_index(drop=True)
    )

    return name_to_ids, detail_df, summary_df


# =========================
# 4) 落盘与派生输出（固定目录，无日期标签）
# =========================
def _write_id_list(filepath: Path, ids: Iterable[str]) -> None:
    """简单的换行分隔写入。"""
    filepath.write_text("\n".join(map(str, ids)) + "\n", encoding="utf-8")


def _ids_for_seedset(seed_df: pd.DataFrame, name_to_ids: Mapping[str, List[str]], seed_set: str) -> List[str]:
    """根据 seed_set（aging / longevity）汇总其所有命中 ID（去重保序）。"""
    names = seed_df.loc[seed_df["seed_set"] == seed_set, "name"].astype(str).unique().tolist()
    ids: List[str] = []
    for n in names:
        ids.extend(name_to_ids.get(n, []))
    return list(dict.fromkeys(ids))


def save_seed_outputs_fixed(
    seed_df: pd.DataFrame,
    name_to_ids: Mapping[str, List[str]],
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """
    将结果保存到固定目录 out_dir（不存在则创建；存在则覆盖同名文件）。
    返回：out_dir。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 多命中/未命中
    name_multiples = {k: v for k, v in name_to_ids.items() if len(v) > 1}
    name_nohits = [k for k, v in name_to_ids.items() if len(v) == 0]

    # 1) seed_master.csv
    seed_master = (
        seed_df[["name", "seed_set", "category", "tier"]]
        .drop_duplicates()
        .assign(
            n_ids=lambda df: df["name"].map(lambda n: len(name_to_ids.get(n, []))),
            ids_preview=lambda df: df["name"].map(lambda n: ";".join(name_to_ids.get(n, [])[:20])),
        )
        .sort_values(["seed_set", "category", "tier", "n_ids", "name"],
                     ascending=[True, True, True, False, True])
    )
    seed_master.to_csv(out_dir / "seed_master.csv", index=False, encoding="utf-8-sig")

    # 2) 明细/汇总/映射
    (out_dir / "name_to_ids.json").write_text(
        json.dumps(name_to_ids, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    if not detail_df.empty:
        detail_df.to_csv(out_dir / "seed_map_detail.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "seed_map_summary.csv", index=False, encoding="utf-8-sig")

    # 3) 多命中/未命中列表
    (out_dir / "name_multiples.json").write_text(
        json.dumps(name_multiples, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    (out_dir / "name_nohits.json").write_text(
        json.dumps(name_nohits, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 4) 三个 ID 清单
    aging_ids = _ids_for_seedset(seed_df, name_to_ids, "aging")
    longevity_ids = _ids_for_seedset(seed_df, name_to_ids, "longevity")
    combined_ids = list(dict.fromkeys(aging_ids + longevity_ids))
    _write_id_list(out_dir / "aging_ids.txt", aging_ids)
    _write_id_list(out_dir / "longevity_ids.txt", longevity_ids)
    _write_id_list(out_dir / "combined_ids.txt", combined_ids)

    # 5) README（简要说明）
    readme = """# HALDxAI Seed Priors

本目录存放“衰老/长寿轴”的先验种子实体映射结果（供 S2S 子网抽取使用）。
- `seed_master.csv`：每个名称一行，含 seed_set/category/tier、命中 ID 数与预览
- `name_to_ids.json`：{name: [entity_id, ...]} 映射
- `seed_map_detail.csv`：每条命中一行（匹配模式、entity_id、类型、权重等）
- `seed_map_summary.csv`：名称粒度统计
- `name_multiples.json` / `name_nohits.json`：复核多/未命中名称
- `aging_ids.txt` / `longevity_ids.txt` / `combined_ids.txt`：纯 ID 清单
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    return out_dir


# =========================
# 5) Notebook 友好的一体化 API（固定目录）
# =========================
def build_seed_priors_fixed(
    seeds_df: pd.DataFrame,
    config: SeedPriorConfig,
    proj_root: Path,
    out_rel_dir: str = "data/HALD-Seed2Subgraph/priors",
) -> dict:
    """
    一次性完成：映射 → DataFrame 返回 → 固定目录落盘（覆盖同名文件）。
    返回：
      {
        "name_to_ids": dict,
        "detail_df": pd.DataFrame,
        "summary_df": pd.DataFrame,
        "aging_ids": list[str],
        "longevity_ids": list[str],
        "combined_ids": list[str],
        "out_dir": Path,
      }
    """
    name_to_ids, detail_df, summary_df = map_names_to_ids(seeds_df, config)
    out_dir = save_seed_outputs_fixed(
        seed_df=seeds_df,
        name_to_ids=name_to_ids,
        detail_df=detail_df,
        summary_df=summary_df,
        out_dir=(proj_root / out_rel_dir).resolve(),
    )
    aging_ids = _ids_for_seedset(seeds_df, name_to_ids, "aging")
    longevity_ids = _ids_for_seedset(seeds_df, name_to_ids, "longevity")
    combined_ids = list(dict.fromkeys(aging_ids + longevity_ids))
    return {
        "name_to_ids": name_to_ids,
        "detail_df": detail_df,
        "summary_df": summary_df,
        "aging_ids": aging_ids,
        "longevity_ids": longevity_ids,
        "combined_ids": combined_ids,
        "out_dir": out_dir,
    }


__all__ = [
    "SeedPriorConfig",
    "clean_name_for_match",
    "_to_key",
    "map_names_to_ids",
    "save_seed_outputs_fixed",
    "build_seed_priors_fixed",
]
