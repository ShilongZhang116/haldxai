#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准化 RNAcentral × Rfam 注释
-------------------------------------------------
生成文件
└─ {project_root}/data/standard/external_db/
   └─ hald_rnacentral__rfam_annotations__std.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


# ════════════════════════════════════════════════
# 1. 单文件解析器
# ════════════════════════════════════════════════
def parse_rnacentral_rfam(tsv_fp: Path) -> pd.DataFrame:
    """
    读取 `rnacentral_rfam_annotations.tsv` 并拆分 URS / TaxID

    源文件（无表头）列序：
        0: URSxxx_taxid
        1: GO ID
        2: Rfam ID
    """
    tmp = pd.read_csv(
        tsv_fp,
        sep="\t",
        header=None,
        names=["urs_taxid", "go_id", "rfam_id"],
        dtype=str,
    )

    # -- 拆成 URS 和 NCBI TaxID
    tmp[["urs_acc", "ncbi_taxid"]] = tmp["urs_taxid"].str.split("_", n=1, expand=True)

    df = (
        tmp[["urs_acc", "ncbi_taxid", "go_id", "rfam_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return df


# ════════════════════════════════════════════════
# 2. 调度函数
# ════════════════════════════════════════════════
def build_rnacentral(
    project_root: Path,
    raw_filename: str = "rnacentral_rfam_annotations.tsv",
    force: bool = False,
):
    """
    解析 RNAcentral × Rfam TSV → 写标准化 CSV
    """
    raw_fp = project_root / "data/bio_corpus/RNAcentral" / raw_filename
    std_dir = project_root / "data/external_db"
    std_dir.mkdir(parents=True, exist_ok=True)
    out_fp = std_dir / "hald_rnacentral__rfam_annotations__std.csv"

    if out_fp.exists() and not force:
        print(f"⚠️  输出已存在: {out_fp}（跳过；如需覆盖加 --force）")
        return

    df = parse_rnacentral_rfam(raw_fp)
    df.to_csv(
        out_fp,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )
    print(f"✅ RNAcentral Rfam 注释标准化完成 → {out_fp}  ({len(df):,} 行)")


# ════════════════════════════════════════════════
# 3. CLI
# ════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser(description="标准化 RNAcentral × Rfam 注释 TSV")
    pa.add_argument("--root", required=True, type=Path, help="HALDxAI-Project 根目录")
    pa.add_argument(
        "--raw-name",
        default="rnacentral_rfam_annotations.tsv",
        help="原始 TSV 文件名（默认同官方）",
    )
    pa.add_argument("--force", action="store_true", help="若已存在则覆盖写出")
    args = pa.parse_args()

    build_rnacentral(args.root, raw_filename=args.raw_name, force=args.force)
