#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准化 Digital Ageing Atlas (DAA)
================================

Notebook / Python
-----------------
from haldxai.enrich.external_db.daa.build_daa import build_daa
build_daa(project_root=Path("/abs/path/to/HALDxAI-Project"), force=False)

CLI（统一入口）
---------------
python -m haldxai.enrich.external_db.cli daa \
       --root /abs/path/to/HALDxAI-Project --force
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from haldxai.enrich.external_db.io_utils import read_tsv_robust  # ← 自动识别制表/逗号

# ════════════════════════════════════════════════════════════════
# 1. 单一数据集的配置
# ════════════════════════════════════════════════════════════════
DAA_CFG = dict(
    raw="data/bio_corpus/DigitalAgeingAtlas/digital_ageing_atlas_data.txt",
    out="hald_daa__digital_ageing_atlas__std.csv",
    cols={
        "Identifier": "daa_id",
        "Change name": "entity_name",
        "Change type": "entity_type",
        "Species": "species",
        "Change gender": "change_gender",
        "Age change starts": "age_change_start",
        "Age change ends": "age_change_end",
        "Description": "description",
        "Tissues": "tissue",
        "Gene": "gene_raw",
        "Properties": "properties",
        "Type of data": "type_of_data",
        "Process measured": "process_measured",
        "Sample size": "sample_size",
        "Method of collection": "method_of_collection",
        "Data transforms": "data_transforms",
        "Percentage change": "percentage_change",
        "P value": "pvalue",
        "Coefficiant": "coefficient",
        "Intercept": "intercept",
        "Relationship parent identifiers": "relationship_parent_daa_id",
        "References (with LibAge reference ID in brackets)": "references_raw",
    },
)

# ════════════════════════════════════════════════════════════════
# 2. gene_raw / references_raw 解析工具
# ════════════════════════════════════════════════════════════════
_GENE_RE = re.compile(r"^\s*([\w\-]+)\s*\((.+?)\)\s*$")
_REF_RE  = re.compile(
    r"(?P<ref_id>\d+):\s*"
    r"(?P<article_authors>.+?)\s+"
    r"\((?P<article_year>\d{4})\)\s*"
    r"\"(?P<article_title>.+?)\"\s*"
    r"(?P<journal_blob>[^()]+?[0-9].+?)\s*"
    r"\((?P<pubmed_id>\d+)\)"
)
_JOURNAL_SPLIT = re.compile(
    r"^\s*(?P<name>[A-Za-z][A-Za-z\.\s\-]+?)\s+(?P<info>[\d][\d().:\-–]+.*)$"
)

def _split_gene(val: str) -> Dict[str, str]:
    """'NT5C2 (5'-nucleotidase, cytosolic II)' → {'gene_symbol': 'NT5C2', 'gene_description': ...}"""
    m = _GENE_RE.match(str(val))
    if m:
        return {"gene_symbol": m.group(1).upper(), "gene_description": m.group(2)}
    return {"gene_symbol": np.nan, "gene_description": np.nan}

def _split_reference(val: str) -> Dict[str, str]:
    m = _REF_RE.match(str(val))
    if not m:
        return {}
    d = m.groupdict()
    blob = d.pop("journal_blob").strip()
    jm = _JOURNAL_SPLIT.match(blob)
    d["journal_name"] = jm.group("name").strip(". ") if jm else blob
    d["journal_info"] = jm.group("info") if jm else np.nan
    return d

# ════════════════════════════════════════════════════════════════
# 3. 核心处理
# ════════════════════════════════════════════════════════════════
def _build_one(project_root: Path, *, force: bool = False) -> None:
    raw_fp = project_root / DAA_CFG["raw"]
    out_fp = project_root / "data/external_db" / DAA_CFG["out"]

    if out_fp.exists() and not force:
        print(f"🟡 {out_fp.name} 已存在（跳过，可用 --force 覆盖）")
        return
    if not raw_fp.exists():
        print(f"❌ 缺失源文件：{raw_fp}")
        return

    df = read_tsv_robust(raw_fp)
    df = df.rename(columns=DAA_CFG["cols"])
    df = df[[c for c in DAA_CFG["cols"].values() if c in df.columns]]

    # ——— gene_raw 拆列 ———
    if "gene_raw" in df.columns:
        gene_df = df["gene_raw"].apply(_split_gene).apply(pd.Series)
        df = pd.concat([df.drop(columns="gene_raw"), gene_df], axis=1)

    # ——— references_raw 拆列 ———
    if "references_raw" in df.columns:
        ref_df = df["references_raw"].apply(_split_reference).apply(pd.Series)
        df = pd.concat([df.drop(columns="references_raw"), ref_df], axis=1)

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(f"✅ {out_fp.name:<55} {len(df):>8,} 行  {df.shape[1]} 列")

# ════════════════════════════════════════════════════════════════
# 4. 对外统一函数
# ════════════════════════════════════════════════════════════════
def build_daa(project_root: Path, *, force: bool = False) -> None:
    """标准化 Digital Ageing Atlas."""
    _build_one(project_root, force=force)

# ════════════════════════════════════════════════════════════════
# 5. CLI 入口
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser(description="标准化 Digital Ageing Atlas (DAA)")
    pa.add_argument("--root", required=True, type=Path, help="HALDxAI-Project 根目录")
    pa.add_argument("--force", action="store_true", help="覆盖已存在 std 文件")
    args = pa.parse_args()

    build_daa(args.root, force=args.force)
