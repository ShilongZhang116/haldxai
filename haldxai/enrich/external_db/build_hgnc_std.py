#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准化 HGNC Gene 信息表
================================

Notebook
--------
from haldxai.enrich.external_db.build_hgnc_std import build_hgnc_std
build_hgnc_std(project_root=Path("/abs/path/to/HALDxAI-Project"), force=False)

CLI
---
python -m haldxai.enrich.external_db.cli ncbi_gene \
       --root /abs/path/to/HALDxAI-Project --force
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import pandas as pd

from haldxai.enrich.external_db.io_utils import read_tsv_robust

# ════════════════════════════════════════════════════════════════
# 1. 配置
# ════════════════════════════════════════════════════════════════
HGNC_CFG: Dict[str, str | Dict] = dict(
    raw="data/bio_corpus/HGNC/hgnc_complete_set.txt",
    out="hald_hgnc__gene__std.csv",
    cols={
        "hgnc_id": "hgnc_id",
        "symbol": "gene_symbol",
        "name": "gene_description",
        "locus_group": "gene_type",
        "locus_type": "locus_type",
        "status": "status",
        "location": "map_location",
        "location_sortable": "location_sortable",
        "alias_symbol": "gene_alias",
        "alias_name": "gene_alias_description",
        "prev_symbol": "prev_symbol",
        "prev_name": "prev_name",
        "gene_group": "gene_group",
        "gene_group_id": "gene_group_id",
        "date_approved_reserved": "date_approved_reserved",
        "date_symbol_changed": "date_symbol_changed",
        "date_name_changed": "date_name_changed",
        "date_modified": "date_modified",
        "entrez_id": "entrez_id",
        "ensembl_gene_id": "ensembl_gene_id",
        "vega_id": "vega_id",
        "ucsc_id": "ucsc_id",
        "ena": "ena",
        "refseq_accession": "refseq_accession",
        "ccds_id": "ccds_id",
        "uniprot_ids": "uniprot_ids",
        "pubmed_id": "pubmed_id",
        "mgd_id": "mgd_id",
        "rgd_id": "rgd_id",
        "lsdb": "lsdb",
        "cosmic": "cosmic",
        "omim_id": "omim_id",
        "mirbase": "mirbase",
        "homeodb": "homeodb",
        "snornabase": "snornabase",
        "bioparadigms_slc": "bioparadigms_slc",
        "orphanet": "orphanet",
        "pseudogene.org": "pseudogene.org",
        "horde_id": "horde_id",
        "merops": "merops",
        "imgt": "imgt",
        "iuphar": "iuphar",
        "kznf_gene_catalog": "kznf_gene_catalog",
        "mamit-trnadb": "mamit-trnadb",
        "cd": "cd",
        "lncrnadb": "lncrnadb",
        "enzyme_id": "enzyme_id",
        "intermediate_filament_db": "intermediate_filament_db",
        "rna_central_id": "rna_central_id",
        "lncipedia": "lncipedia",
        "gtrnadb": "gtrnadb",
        "agr": "agr",
        "mane_select": "mane_select",
        "gencc": "gencc",
    },
)

# ════════════════════════════════════════════════════════════════
# 2. 核心函数
# ════════════════════════════════════════════════════════════════
def _build_one(project_root: Path, *, force: bool = False) -> None:

    raw_fp: Path = project_root / HGNC_CFG["raw"]            # type: ignore
    out_fp: Path = project_root / "data/external_db" / HGNC_CFG["out"]  # type: ignore

    if out_fp.exists() and not force:
        print(f"🟡 {out_fp.name} 已存在（跳过，可用 --force 覆盖）")
        return
    if not raw_fp.exists():
        print(f"❌ 缺失源文件：{raw_fp}")
        return

    # 1) 读取
    df = pd.read_csv(raw_fp, encoding='utf-8-sig', sep='\t', low_memory=False)

    # 2) 列重命名 & 裁剪
    col_map: Dict[str, str] = HGNC_CFG["cols"]  # type: ignore
    df = df.rename(columns=col_map)
    keep_cols: List[str] = list(col_map.values())
    df = df[[c for c in keep_cols if c in df.columns]]

    # 3) 保存
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(f"✅ {out_fp.name:<40} {len(df):>8,} 行  {df.shape[1]} 列")

# ════════════════════════════════════════════════════════════════
# 3. 对外 API
# ════════════════════════════════════════════════════════════════
def build_hgnc_gene(project_root: Path, *, force: bool = False) -> None:
    """标准化 HGNC表 为 hald_hgnc__std.csv"""
    _build_one(project_root, force=force)

# ════════════════════════════════════════════════════════════════
# 4. CLI 入口
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser(description="标准化 HGNC")
    pa.add_argument("--root", required=True, type=Path, help="HALDxAI-Project 根目录")
    pa.add_argument("--force", action="store_true", help="覆盖已存在 std 文件")
    args = pa.parse_args()

    build_hgnc_gene(args.root, force=args.force)
