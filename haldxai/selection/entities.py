# -*- coding: utf-8 -*-
"""
HALD · 高可信实体筛选模块（通用命名）
===================================
输入：aggregation 产物 entity_type_weights_full.parquet（或 DataFrame）
关键列要求：
  entity_id, entity_type, evidence_count, sources_unique, weight_sum, weight_norm, rank_in_entity

核心函数：
- select_entities_high_confidence(...)      # 多类型一起筛
- select_type_high_confidence(..., etype=)  # 单一类型（BMC/AAI/…）通用

兼容别名（可选）：
- select_bmc_high_confidence -> 调用 select_type_high_confidence(..., etype="BMC")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, List, Union
import json
import pandas as pd
import numpy as np

REQUIRED_COLS = {
    "entity_id", "entity_type", "evidence_count",
    "sources_unique", "weight_sum", "weight_norm", "rank_in_entity"
}

@dataclass
class SelectionParams:
    """筛选参数"""
    entity_types: Iterable[str]                      # 需要筛选的实体类型列表，如 ["BMC"] or ["BMC","AAI"]
    top_primary_only: bool = True                    # 仅主类型（rank_in_entity==1）
    min_weight_norm: Optional[float] = 0.60          # 固定阈值；设为 None 则按分位数
    percentile_q: float = 0.85                       # 当 min_weight_norm=None 时使用该分位数
    min_evidence_count: int = 2                      # 证据行数下限
    min_sources_unique: int = 1                      # 不同来源命中下限
    case_insensitive_type: bool = False              # 实体类型是否大小写不敏感
    id2name_path: Optional[Union[str, Path]] = None  # 可选：id→name 映射
    output_dir: Optional[Union[str, Path]] = None    # 可选：落盘目录
    save_seed_json: bool = True                      # 是否输出 seeds JSON（每类型一份 + 合并一份）
    export_cols: List[str] = field(default_factory=lambda: [
        "entity_id", "name", "entity_type", "weight_norm",
        "weight_sum", "evidence_count", "sources_unique", "rank_in_entity"
    ])

def _load_weights(input_obj: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """从 parquet/csv/DataFrame 读取，并规范必要列类型"""
    if isinstance(input_obj, pd.DataFrame):
        df = input_obj.copy()
    else:
        p = Path(input_obj)
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p, dtype=str, low_memory=False)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    df["entity_id"] = df["entity_id"].astype(str)
    df["entity_type"] = df["entity_type"].astype(str)
    for c in ("evidence_count", "sources_unique"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in ("weight_sum", "weight_norm"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    if "rank_in_entity" in df.columns:
        df["rank_in_entity"] = pd.to_numeric(df["rank_in_entity"], errors="coerce").fillna(9999).astype(int)
    return df

def _attach_names(df: pd.DataFrame, id2name_path: Optional[Union[str, Path]]) -> pd.DataFrame:
    """合并 id→name（可选）"""
    out = df.copy()
    out["name"] = ""
    if not id2name_path:
        return out
    p = Path(id2name_path)
    if not p.exists():
        return out
    try:
        id2name = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        id2name = {}
    out["name"] = out["entity_id"].map(lambda x: id2name.get(str(x), ""))
    return out

def _filter_one_type(df_all: pd.DataFrame, etype: str, params: SelectionParams) -> pd.DataFrame:
    """对单个实体类型进行筛选"""
    if params.case_insensitive_type:
        mask_type = df_all["entity_type"].str.lower() == etype.lower()
    else:
        mask_type = df_all["entity_type"] == etype

    sub = df_all[mask_type].copy()
    if sub.empty:
        return sub

    if params.top_primary_only and "rank_in_entity" in sub.columns:
        sub = sub[sub["rank_in_entity"] == 1].copy()

    # 阈值（固定或分位）
    thr = float(sub["weight_norm"].quantile(params.percentile_q)) if params.min_weight_norm is None \
          else float(params.min_weight_norm)

    mask = (
        (sub["weight_norm"] >= thr) &
        (sub["evidence_count"] >= int(params.min_evidence_count)) &
        (sub["sources_unique"] >= int(params.min_sources_unique))
    )
    sub = sub[mask].copy()

    # 排序
    sort_cols, ascending = ["weight_norm"], [False]
    if "weight_sum" in sub.columns:
        sort_cols.append("weight_sum"); ascending.append(False)
    sort_cols.append("evidence_count"); ascending.append(False)
    sub = sub.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    sub["entity_type"] = etype  # 统一输出类型名（大小写一致）
    return sub

def select_entities_high_confidence(
    weights_input: Union[str, Path, pd.DataFrame],
    params: SelectionParams
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    通用入口（多类型）：按多个实体类型筛选高可信实体。
    返回：
      - combined_df：合并后的 DataFrame（包含所有类型）
      - seeds_dict ：{etype: [entity_id,...]}
    若 params.output_dir 提供，则自动落盘每类型 CSV/JSON 及合并版。
    """
    df_all = _load_weights(weights_input)
    results = []
    seeds_dict: Dict[str, List[str]] = {}

    for etype in params.entity_types:
        sub = _filter_one_type(df_all, etype, params)
        if sub.empty:
            sub = pd.DataFrame(columns=list(df_all.columns))

        sub = _attach_names(sub, params.id2name_path)

        cols_out = [c for c in params.export_cols if c in sub.columns]
        if cols_out:
            sub = sub[cols_out + [c for c in ("entity_type",) if c not in cols_out]]

        results.append(sub)
        seeds_dict[etype] = sub["entity_id"].astype(str).tolist() if not sub.empty else []

        if params.output_dir:
            out_dir = Path(params.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            basename = etype.replace("/", "_").replace("\\", "_")
            sub.to_csv(out_dir / f"{basename.lower()}_high_confidence_entities.csv",
                       index=False, encoding="utf-8-sig")
            if params.save_seed_json:
                (out_dir / f"{basename.lower()}_seed_ids.json").write_text(
                    json.dumps(seeds_dict[etype], ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )

    combined = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=list(df_all.columns))
    if params.output_dir:
        out_dir = Path(params.output_dir)
        combined.to_csv(out_dir / "selected_entities_all.csv", index=False, encoding="utf-8-sig")
        if params.save_seed_json:
            all_seeds = sorted({eid for vs in seeds_dict.values() for eid in vs})
            (out_dir / "seed_ids_all.json").write_text(
                json.dumps(all_seeds, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    return combined, seeds_dict

def select_type_high_confidence(
    weights_input: Union[str, Path, pd.DataFrame],
    *,
    etype: str,
    output_dir: Optional[Union[str, Path]] = None,
    id2name_path: Optional[Union[str, Path]] = None,
    min_weight_norm: Optional[float] = 0.60,
    percentile_q: float = 0.85,
    min_evidence_count: int = 2,
    min_sources_unique: int = 1,
    top_primary_only: bool = True,
    case_insensitive_type: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    单一类型通用封装：BMC / AAI / ... 都用这个。
    """
    params = SelectionParams(
        entity_types=[etype],
        top_primary_only=top_primary_only,
        min_weight_norm=min_weight_norm,
        percentile_q=percentile_q,
        min_evidence_count=min_evidence_count,
        min_sources_unique=min_sources_unique,
        case_insensitive_type=case_insensitive_type,
        id2name_path=id2name_path,
        output_dir=output_dir
    )
    df, seeds = select_entities_high_confidence(weights_input, params)
    return df, seeds.get(etype, [])

# --- 兼容旧名（可保留一段时间；也可以删除） ---
def select_bmc_high_confidence(
    weights_input: Union[str, Path, pd.DataFrame],
    **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """兼容旧名：等价于 select_type_high_confidence(..., etype='BMC')"""
    return select_type_high_confidence(weights_input, etype="BMC", **kwargs)
