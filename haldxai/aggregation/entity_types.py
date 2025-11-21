# haldxai/weights/entity_types.py
# -*- coding: utf-8 -*-
"""
HALD · entity_types 按“来源(source)”加权聚合
================================================
功能
----
给 entity_types（至少含: entity_id, entity_type, source）按来源映射权重，
对 (entity_id, entity_type) 聚合，并在实体内做归一化，生成“类型可信度权重”。

主要入口
--------
- infer_source_weight(source_str, source_wt=..., source_alias=..., default_wt=0.5)
- aggregate_entity_type_weights(df, source_wt=..., source_alias=..., default_wt=0.5)
- compute_entity_type_weights(input_path_or_df, save_dir=None, ...)

输出列说明（聚合结果）
--------------------
- entity_id                : 实体统一 ID
- entity_type              : 类型
- evidence_count           : 此 (id,type) 的原始记录条数
- sources_unique           : 命中来源键（canonical key）去重数
- weight_sum               : 行级权重求和
- weight_mean              : 行级权重均值
- weight_max               : 行级权重最大值
- default_hits             : 使用默认权重的行数
- any_default              : 是否出现默认权重
- weight_total_entity      : 该 entity_id 下所有类型的 weight_sum 之和
- weight_norm              : 在实体内部归一化后的权重（和为 1）
- rank_in_entity           : 在实体内按 weight_sum 排名（1 为主类型）

Notebook 中如何调用
from pathlib import Path
import pandas as pd

# 1) 导入模块
from haldxai.weights.entity_types import compute_entity_type_weights

ROOT = Path(r"G:\Project\HALDxAI-Suite\HALDxAI-Project")
ENTITY_TYPES_CSV = ROOT / r"data\database\entity_types.csv"
OUT_DIR = ROOT / r"cache\weights"

# 2) 一行搞定：计算 + 落盘（parquet / topK / 覆盖率）
weights_df = compute_entity_type_weights(
    ENTITY_TYPES_CSV,
    save_dir=OUT_DIR,          # 想只计算不落盘，就设为 None
    topk_per_entity=3          # 导出每个实体 Top-3 类型（可改为 1/2/…）
)

# 3) 取每个实体的主类型（rank_in_entity == 1）
primary = weights_df[weights_df["rank_in_entity"] == 1][
    ["entity_id", "entity_type", "weight_norm"]
]

# 4) 若只想计算不落盘：
# weights_df = compute_entity_type_weights(ENTITY_TYPES_CSV, save_dir=None)

如果要自定义权重或别名
from haldxai.weights.entity_types import compute_entity_type_weights, DEFAULT_SOURCE_WT

MY_WT = dict(DEFAULT_SOURCE_WT)
MY_WT["AgingRelated-DeepSeekR1-32B"] = 0.65   # 举例微调

weights_df = compute_entity_type_weights(
    ENTITY_TYPES_CSV,
    save_dir=OUT_DIR,
    source_wt=MY_WT,            # 用你的权重
    source_alias={"uniprot":"uniport"},  # 可加/改别名
    default_wt=0.45
)

"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Union
import re
import json
import numpy as np
import pandas as pd

# ---------------- 默认权重与别名（可在函数参数里覆盖） ----------------
DEFAULT_SOURCE_WT: Dict[str, float] = {
    'ageanno': 1.0, 'agingatlas': 1.0, 'ctd': 1.0, 'daa': 1.0, 'hagr': 1.0, 'hall': 1.0,
    'hetionet': 1.0, 'hgnc': 1.0, 'icd10': 1.0, 'miRBase': 1.0, 'ncbi': 1.0, 'opengenes': 1.0,
    'primekg': 1.0, 'umls': 1.0, 'uniport': 1.0,
    'JCRQ1-IF10-DeepSeekV3': 0.8, 'AgingRelated-DeepSeekV3': 0.8,
    'AgingRelated-DeepSeekR1-32B': 0.7, 'JCRQ1-IF10-DeepSeekR1-32B': 0.7,
    'AgingRelated-DeepSeekR1-7B': 0.6, 'JCRQ1-IF10-DeepSeekR1-7B': 0.6,
    'bert_model_prediction': 0.4,
    'en_ner_bc5cdr_md': 0.3, 'en_ner_bionlp13cg_md': 0.3, 'en_ner_jnlpba_md': 0.3,
}

DEFAULT_SOURCE_ALIAS: Dict[str, str] = {
    "uniprot": "uniport",   # 常见拼写
    "mirbase": "miRBase",
    "ernie": "ERINE",       # 预留示例
}

DEFAULT_WT: float = 0.5     # 未命中来源键时的默认权重


def _build_lookup(source_wt: Dict[str, float]) -> Tuple[List[str], Dict[str, float]]:
    """构造小写键查找表与按长度降序的键列表，便于“包含匹配”加速。"""
    wt_lower = {k.lower(): v for k, v in source_wt.items()}
    keys_lower = sorted(list(wt_lower.keys()), key=len, reverse=True)
    return keys_lower, wt_lower


def _canon_token(tok: str, source_alias: Dict[str, str]) -> str:
    """
    单 token 规范化：
      - 去空白/符号、小写
      - 处理 '__' 前缀（如 ageanno__xxx → ageanno）
      - 处理 'source:xxx' 形式（取冒号后部分）
      - 应用别名映射
    """
    s = (tok or "").strip()
    if not s:
        return ""
    s = s.replace(" ", "")
    s_low = s.lower()
    if "__" in s_low:
        s_low = s_low.split("__", 1)[0]
    if ":" in s_low:
        s_low = s_low.split(":", 1)[-1]
    s_low = s_low.strip("[]()")
    # 别名
    for k, v in source_alias.items():
        if s_low == k.lower():
            return v.lower()
    return s_low


def infer_source_weight(
    source_str: str,
    *,
    source_wt: Dict[str, float] = DEFAULT_SOURCE_WT,
    source_alias: Dict[str, str] = DEFAULT_SOURCE_ALIAS,
    default_wt: float = DEFAULT_WT,
) -> Tuple[float, List[str], bool]:
    """
    将一条 `source` 字段映射为行级权重。
    返回:
      - row_weight: float       （命中多个来源时取“最大权重”；也可按需改成均值/求和）
      - hit_keys: List[str]     （命中的 canonical key（小写）列表）
      - used_default: bool      （是否使用默认权重）
    """
    if not isinstance(source_str, str) or not source_str.strip():
        return default_wt, [], True

    keys_lower, wt_lower = _build_lookup(source_wt)

    s = source_str.strip()
    # 拆分潜在的多个来源
    raw_tokens = re.split(r"[;,|]+", s)
    # 额外加一份去 '__' 前缀的 token 候选
    more = []
    for t in raw_tokens:
        tl = t.lower()
        if "__" in tl:
            more.append(tl.split("__", 1)[0])
    tokens = list({*raw_tokens, *more})

    cand = set()
    for t in tokens:
        ctok = _canon_token(t, source_alias)
        if not ctok:
            continue
        # 等值命中
        if ctok in wt_lower:
            cand.add(ctok)
            continue
        # 宽松包含匹配（长键优先）
        for key_low in keys_lower:
            if key_low in ctok:
                cand.add(key_low)
                break

    if cand:
        max_w = max(wt_lower[c] for c in cand)
        return float(max_w), sorted(list(cand)), False
    return float(default_wt), [], True


def aggregate_entity_type_weights(
    df: pd.DataFrame,
    *,
    source_wt: Dict[str, float] = DEFAULT_SOURCE_WT,
    source_alias: Dict[str, str] = DEFAULT_SOURCE_ALIAS,
    default_wt: float = DEFAULT_WT,
) -> pd.DataFrame:
    """
    对 entity_types DataFrame（至少含列: entity_id, entity_type, source）进行加权聚合。

    参数
    ----
    df : 包含 entity_id / entity_type / source 的 DataFrame
    source_wt / source_alias / default_wt : 权重表、别名映射、默认权重（可覆盖默认）

    返回
    ----
    聚合结果 DataFrame（列见模块头部说明）
    """
    required = {"entity_id", "entity_type", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"输入缺少必要列: {missing}")

    x = df.dropna(subset=["entity_id", "entity_type"]).copy()
    x["entity_id"] = x["entity_id"].astype(str)

    # 行级权重映射
    wt_res = x["source"].map(lambda s: infer_source_weight(
        s, source_wt=source_wt, source_alias=source_alias, default_wt=default_wt
    ))
    x["row_weight"]      = wt_res.map(lambda r: r[0])
    x["hit_keys"]        = wt_res.map(lambda r: ";".join(r[1]) if r[1] else "")
    x["used_default_wt"] = wt_res.map(lambda r: r[2])

    # (entity_id, entity_type) 聚合
    grp = x.groupby(["entity_id", "entity_type"], as_index=False).agg(
        evidence_count = ("source", "size"),
        sources_unique = ("hit_keys", lambda s: len(set(k for v in s for k in (v.split(";") if v else [])))),
        weight_sum     = ("row_weight", "sum"),
        weight_mean    = ("row_weight", "mean"),
        weight_max     = ("row_weight", "max"),
        default_hits   = ("used_default_wt", "sum"),
        any_default    = ("used_default_wt", "any"),
    )

    # 在实体内部归一化（Sum=1）
    grp["weight_total_entity"] = grp.groupby("entity_id")["weight_sum"].transform("sum")
    grp["weight_norm"] = grp["weight_sum"] / grp["weight_total_entity"].replace({0: np.nan})
    grp["weight_norm"] = grp["weight_norm"].fillna(0.0)

    # 排名（实体内）
    grp["rank_in_entity"] = grp.groupby("entity_id")["weight_sum"] \
                               .rank(method="first", ascending=False).astype(int)

    # 排序便于查看
    grp = grp.sort_values(["entity_id", "weight_sum"], ascending=[True, False]).reset_index(drop=True)
    return grp


def source_coverage_report(
    df_row_level: pd.DataFrame,
    *,
    source_wt: Dict[str, float] = DEFAULT_SOURCE_WT,
    source_alias: Dict[str, str] = DEFAULT_SOURCE_ALIAS,
    default_wt: float = DEFAULT_WT,
) -> pd.DataFrame:
    """
    行级覆盖率报告：统计命中哪些来源键、默认权重占比等。
    传入 **原始行级 DataFrame**（至少含 source 列）。
    """
    if "source" not in df_row_level.columns:
        raise ValueError("缺少列 'source'。")

    # 计算行级映射
    tmp = df_row_level.copy()
    res = tmp["source"].map(lambda s: infer_source_weight(
        s, source_wt=source_wt, source_alias=source_alias, default_wt=default_wt
    ))
    tmp["row_weight"] = res.map(lambda r: r[0])
    tmp["hit_keys"]   = res.map(lambda r: ";".join(r[1]) if r[1] else "")

    tmp["hit_keys_list"] = tmp["hit_keys"].apply(lambda s: [k for k in s.split(";") if k])
    exploded = tmp.explode("hit_keys_list")
    cov = exploded.groupby("hit_keys_list", dropna=False).agg(
        rows=("hit_keys_list", "size"),
        mean_row_wt=("row_weight", "mean")
    ).reset_index().rename(columns={"hit_keys_list": "hit_key"})

    # 映射到 canonical key 的权重
    _, wt_lower = _build_lookup(source_wt)
    cov["mapped_weight"] = cov["hit_key"].map(wt_lower).astype(float)
    cov = cov.sort_values("rows", ascending=False).reset_index(drop=True)
    return cov


def compute_entity_type_weights(
    input_path_or_df: Union[str, Path, pd.DataFrame],
    *,
    save_dir: Union[str, Path, None] = None,
    source_wt: Dict[str, float] = DEFAULT_SOURCE_WT,
    source_alias: Dict[str, str] = DEFAULT_SOURCE_ALIAS,
    default_wt: float = DEFAULT_WT,
    topk_per_entity: int = 3,
    read_usecols: tuple = ("entity_id", "entity_type", "source"),
) -> pd.DataFrame:
    """
    从 CSV/Parquet/现成 DF 计算聚合权重；可选择落盘。

    参数
    ----
    input_path_or_df : 路径（str/Path，csv/parquet）或已加载的 DataFrame
    save_dir         : 若不为 None，则将以下文件落盘：
                       - entity_type_weights_full.parquet
                       - entity_type_topK.csv
                       - source_weight_coverage.csv
    其余参数见上。

    返回
    ----
    聚合结果 DataFrame
    """
    # 1) 读取
    if isinstance(input_path_or_df, (str, Path)):
        p = Path(input_path_or_df)
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p, columns=list(read_usecols))
        else:
            df = pd.read_csv(p, dtype=str, usecols=lambda c: c in set(read_usecols), low_memory=False)
    elif isinstance(input_path_or_df, pd.DataFrame):
        df = input_path_or_df.copy()
        # 只保留必要列
        df = df[[c for c in read_usecols if c in df.columns]]
    else:
        raise TypeError("input_path_or_df 需要是路径(str/Path)或 pandas.DataFrame")

    # 2) 计算
    agg = aggregate_entity_type_weights(
        df, source_wt=source_wt, source_alias=source_alias, default_wt=default_wt
    )

    # 3) （可选）落盘
    if save_dir is not None:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 全量聚合
        agg.to_parquet(out / "entity_type_weights_full.parquet", index=False)

        # Top-K
        topk = agg[agg["rank_in_entity"] <= int(topk_per_entity)].copy()
        topk.to_csv(out / f"entity_type_top{int(topk_per_entity)}.csv", index=False, encoding="utf-8-sig")

        # 覆盖率
        cov = source_coverage_report(
            df, source_wt=source_wt, source_alias=source_alias, default_wt=default_wt
        )
        cov.to_csv(out / "source_weight_coverage.csv", index=False, encoding="utf-8-sig")

    return agg
