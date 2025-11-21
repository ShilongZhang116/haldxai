#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from haldxai.enrich.tables.loader import load_name2id, save_name2id
from haldxai.enrich.tables.utils import alloc_id

def _clean_source(src: str) -> str:
    """
    统一压缩 source_file:
    ▸ hald_ageanno__pathways__std.csv  →  ageanno__pathways
    """
    src = Path(src).name                       # 只保留文件名
    src = re.sub(r"^hald_", "", src)
    src = re.sub(r"__std\.csv$", "", src)
    return src

def build_entity_catalog_ext(
        project_root: Path,
        df_ext_nodes: pd.DataFrame,
        *,
        force: bool = False,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df_ext_nodes : DataFrame
        必须包含列 `entity_name, entity_type, primary_info, extra_json, source_file`
        （由 ext_collect.py 输出）
    name2id : dict | Path
        名称 → HALD 实体 ID 的映射，或映射 JSON 路径
    lowercase_key : bool , default=True
        是否对比时统一转小写（推荐）

    Returns
    -------
    DataFrame
        扩展目录，字段见模块顶部说明
    """
    db_dir      = project_root / "data/database"
    db_dir.mkdir(parents=True, exist_ok=True)

    output_csv = db_dir / "entity_catalog_ext.csv"

    if output_csv.exists() and not force:
        print(f"🟡 entity_catalog_ext.csv 已存在（跳过）。pass `force=True` 以重新生成。")
        return pd.read_csv(output_csv)

    print("▶ 构建 entity_catalog_ext.csv …")

    # ── 0) 映射加载 ────────────────────────────────
    name2id = load_name2id(project_root)

    # ── 1) 字段规范 & 轻度清洗 ──────────────────────
    need_cols = ["entity_name", "entity_type", "primary_info", "extra_json", "source_file"]
    miss = [c for c in need_cols if c not in df_ext_nodes.columns]
    if miss:
        raise ValueError(f"df_ext_nodes 缺少列: {miss}")

    df = (
        df_ext_nodes[need_cols]
        .rename(columns={"source_file": "source"})
        .copy()
    )
    df["source"] = df["source"].map(_clean_source)

    # ── 2) 生成 entity_id ──────────────────────────
    df["entity_id"] = df["entity_name"].apply(lambda n: alloc_id(name2id, n))

    # ── 3) 自增主键 ────────────────────────────────
    df.insert(0, "ext_pk", range(1, len(df) + 1))

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✓ entity_catalog_ext 写出 {len(df):,} 行 → {output_csv}")

    save_name2id(project_root, name2id)  # ② 把可能新增的映射落盘
    print("✓ name2id.json 已更新，当前条数 =", len(name2id))

    return df