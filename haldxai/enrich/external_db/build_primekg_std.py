#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准化 PrimeKG ➜ HALD _std 表
--------------------------------
生成 4 个文件（UTF-8-SIG、逗号分隔）

    • hald_primekg__kg_relation__std.csv
    • hald_primekg__node__std.csv
    • hald_primekg__disease_features__std.csv
    • hald_primekg__drug_features__std.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


# ════════════════════════════════════════════════
# 1. 解析函数
# ════════════════════════════════════════════════
def read_kg_relation(fp: Path) -> pd.DataFrame:
    """kg.csv → 关系表 DataFrame"""
    col_map = {
        "relation": "relation_type",
        "display_relation": "display_relation",
        "x_index": "source_entity_index",
        "x_id": "source_entity_id",
        "x_type": "source_entity_type",
        "x_name": "source_entity_name",
        "x_source": "source_entity_data_source",
        "y_index": "target_entity_index",
        "y_id": "target_entity_id",
        "y_type": "target_entity_type",
        "y_name": "target_entity_name",
        "y_source": "target_entity_data_source",
    }
    df = (
        pd.read_csv(fp, low_memory=False)
        .rename(columns=col_map)
        .loc[:, list(col_map.values())]  # 只保留映射列并按顺序
    )
    return df


def read_nodes(fp: Path) -> pd.DataFrame:
    """nodes.csv → 节点主表"""
    col_map = {
        "node_index": "entity_index",
        "node_id": "entity_id",
        "node_type": "entity_type",
        "node_name": "entity_name",
        "node_source": "entity_data_source",
    }
    df = (
        pd.read_csv(fp, low_memory=False)
        .rename(columns=col_map)
        .loc[:, list(col_map.values())]
    )
    return df


def _attach_name(df_feat: pd.DataFrame, idx2name: dict[int, str]) -> pd.DataFrame:
    """疾病 / 药物特征表补充 entity_name"""
    if "node_index" in df_feat.columns:
        df_feat = df_feat.rename(columns={"node_index": "entity_index"})
    if "entity_index" not in df_feat.columns:
        return df_feat
    df_feat["entity_name"] = df_feat["entity_index"].map(idx2name).fillna("")
    return df_feat


# ════════════════════════════════════════════════
# 2. 调度函数
# ════════════════════════════════════════════════
def build_primekg(
    project_root: Path,
    *,
    force: bool = False,
    kg_file: str = "kg.csv",
    node_file: str = "nodes.csv",
    disease_file: str = "disease_features.csv",
    drug_file: str = "drug_features.csv",
):
    """
    解析 PrimeKG 多个 csv 并写出标准化结果
    """
    raw_dir = project_root / "data/bio_corpus/PrimeKG/data"
    std_dir = project_root / "data/external_db"
    std_dir.mkdir(parents=True, exist_ok=True)

    out_rel = std_dir / "hald_primekg__kg_relation__std.csv"
    out_node = std_dir / "hald_primekg__node__std.csv"
    out_dis = std_dir / "hald_primekg__disease_features__std.csv"
    out_drug = std_dir / "hald_primekg__drug_features__std.csv"

    if (
        not force
        and out_rel.exists()
        and out_node.exists()
        and out_dis.exists()
        and out_drug.exists()
    ):
        print("⚠️  标准化文件已存在；使用 --force 覆盖")
        return

    # ------------ 读取并转换 -----------------
    print("▶ Loading PrimeKG csv …")
    df_rel = read_kg_relation(raw_dir / kg_file)
    df_node = read_nodes(raw_dir / node_file)

    idx2name = dict(zip(df_node["entity_index"], df_node["entity_name"]))

    df_disease = pd.read_csv(raw_dir / disease_file, low_memory=False)
    df_disease = _attach_name(df_disease, idx2name)

    df_drug = pd.read_csv(raw_dir / drug_file, low_memory=False)
    df_drug = _attach_name(df_drug, idx2name)

    # ------------ 写出 -----------------------
    def _save(df: pd.DataFrame, fp: Path):
        df.to_csv(
            fp,
            index=False,
            encoding="utf-8-sig",
            quoting=csv.QUOTE_MINIMAL,
        )

    _save(df_rel, out_rel)
    _save(df_node, out_node)
    _save(df_disease, out_dis)
    _save(df_drug, out_drug)

    print("✅ PrimeKG 标准化完成：")
    print(f"   {out_rel.name}   ({len(df_rel):,} rows)")
    print(f"   {out_node.name}  ({len(df_node):,} rows)")
    print(f"   {out_dis.name}   ({len(df_disease):,} rows)")
    print(f"   {out_drug.name}  ({len(df_drug):,} rows)")


# ════════════════════════════════════════════════
# 3. CLI
# ════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser(description="标准化 PrimeKG csv 文件")
    pa.add_argument(
        "--root",
        required=True,
        type=Path,
        help="HALDxAI-Project 根目录",
    )
    pa.add_argument("--force", action="store_true", help="覆盖已存在输出")
    pa.add_argument("--kg-file", default="kg.csv", help="kg.csv 文件名")
    pa.add_argument("--node-file", default="nodes.csv", help="nodes.csv 文件名")
    pa.add_argument(
        "--disease-file",
        default="disease_features.csv",
        help="disease_features.csv 文件名",
    )
    pa.add_argument(
        "--drug-file",
        default="drug_features.csv",
        help="drug_features.csv 文件名",
    )
    args = pa.parse_args()

    build_primekg(
        args.root,
        force=args.force,
        kg_file=args.kg_file,
        node_file=args.node_file,
        disease_file=args.disease_file,
        drug_file=args.drug_file,
    )
