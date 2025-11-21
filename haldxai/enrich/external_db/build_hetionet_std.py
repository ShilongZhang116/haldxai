#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准化 Hetionet-v1.0 JSON ➜ CSV
--------------------------------
生成
    • hald_hetionet__node__std.csv
    • hald_hetionet__rel__std.csv
"""

from __future__ import annotations

import json
import csv
from pathlib import Path

import pandas as pd


# ════════════════════════════════════════════════
# 1. 解析函数
# ════════════════════════════════════════════════
def parse_hetionet_json(json_fp: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    读取 Hetionet v1.0 JSON，返回 (节点表, 关系表)
    """

    with open(json_fp, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    nodes: list[dict] = data["nodes"]
    edges: list[dict] = data.get("edges") or data.get("relations")

    # —— 1. 节点 —— --------------------------------------------------
    node_rows: list[dict] = []
    for n in nodes:
        node_rows.append(
            dict(
                entity_name=n.get("name", ""),
                entity_type=n["kind"],
                entity_ref_id=str(n["identifier"]),
                data_source=n["data"].get("source", ""),
                data_source_url=n["data"].get("url", ""),
                src_db="hetionet",
            )
        )
    df_node = pd.DataFrame(node_rows)

    # —— 2. 快速 name 查表（万一 source/target 缺 name） ——
    ref2name = dict(zip(df_node["entity_ref_id"], df_node["entity_name"]))

    def _safe_name(ref_id: str, kind: str) -> str:
        name = ref2name.get(ref_id)
        return name if name else f"{kind}:{ref_id}"

    # —— 3. 关系 —— --------------------------------------------------
    rel_rows: list[dict] = []
    for e in edges:
        s_kind, s_id = e["source_id"]
        t_kind, t_id = e["target_id"]
        s_id = str(s_id)
        t_id = str(t_id)

        rel_rows.append(
            dict(
                source_entity_type=s_kind,
                source_entity_ref_id=s_id,
                source_entity_name=_safe_name(s_id, s_kind),
                target_entity_type=t_kind,
                target_entity_ref_id=t_id,
                target_entity_name=_safe_name(t_id, t_kind),
                rel_type=e["kind"],
                direction=e.get("direction", "both"),
                data_source=e["data"].get("source", ""),
                unbiased=e["data"].get("unbiased", ""),
                src_db="hetionet",
            )
        )
    df_rel = pd.DataFrame(rel_rows)

    return df_node, df_rel


# ════════════════════════════════════════════════
# 2. 调度函数
# ════════════════════════════════════════════════
def build_hetionet(
    project_root: Path,
    json_name: str = "hetionet-v1.0.json",
    force: bool = False,
):
    """
    解析 hetionet JSON 并写标准化 CSV

    Parameters
    ----------
    project_root : Path
        HALDxAI 项目根目录
    json_name : str
        data/bio_corpus/hetionet/hetnet/json/ 下的文件名
    force : bool
        已存在输出时是否覆盖
    """
    raw_fp = (
        project_root
        / "data/bio_corpus/hetionet/hetnet/json"
        / json_name
    )
    std_dir = project_root / "data/external_db"
    std_dir.mkdir(parents=True, exist_ok=True)

    out_node = std_dir / "hald_hetionet__node__std.csv"
    out_rel = std_dir / "hald_hetionet__rel__std.csv"

    if not force and out_node.exists() and out_rel.exists():
        print("⚠️  标准化文件已存在；使用 --force 覆盖")
        return

    print(f"▶ 解析 {raw_fp.name} …")
    df_node, df_rel = parse_hetionet_json(raw_fp)

    df_node.to_csv(
        out_node, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL
    )
    df_rel.to_csv(
        out_rel, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL
    )

    print("✅ 写出完成：")
    print(f"   {out_node.name}  ({len(df_node):,} rows)")
    print(f"   {out_rel.name}   ({len(df_rel):,} rows)")


# ════════════════════════════════════════════════
# 3. CLI
# ════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser(description="标准化 Hetionet v1.0 JSON")
    pa.add_argument(
        "--root",
        required=True,
        type=Path,
        help="HALDxAI-Project 根目录",
    )
    pa.add_argument(
        "--json-name",
        default="hetionet-v1.0.json",
        help="hetnet/json 下原始文件名",
    )
    pa.add_argument("--force", action="store_true", help="覆盖已存在输出")
    args = pa.parse_args()

    build_hetionet(args.root, json_name=args.json_name, force=args.force)
