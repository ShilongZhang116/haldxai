#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准化 miRBase 信息表
================================

Notebook
--------
from haldxai.enrich.external_db.build_mirbase_std import build_mirbase_std
build_mirbase_std(project_root=Path("/abs/path/to/HALDxAI-Project"), force=False)

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
MIRBASE_CFG: Dict[str, str | Dict] = dict(
    raw="data/bio_corpus/miRBase/miRNA.csv",
    out="hald_miRBase__miRBase__std.csv",
    cols={
        "Accession": "mirbase_accession",
        "ID": "mir_id",
        "Status": "status",
        "Sequence": "sequence",
        "Mature1_Acc": "mature1_accession",
        "Mature1_ID": "mature1_id",
        "Mature1_Seq": "mature1_sequence",
        "Mature2_Acc": "mature2_accession",
        "Mature2_ID": "mature2_id",
        "Mature2_Seq": "mature2_sequence",
    },
)

# ════════════════════════════════════════════════════════════════
# 2. 核心函数
# ════════════════════════════════════════════════════════════════
def _build_one(project_root: Path, *, force: bool = False) -> None:

    raw_fp: Path = project_root / MIRBASE_CFG["raw"]            # type: ignore
    out_fp: Path = project_root / "data/external_db" / MIRBASE_CFG["out"]  # type: ignore

    if out_fp.exists() and not force:
        print(f"🟡 {out_fp.name} 已存在（跳过，可用 --force 覆盖）")
        return
    if not raw_fp.exists():
        print(f"❌ 缺失源文件：{raw_fp}")
        return

    # 1) 读取
    df = read_tsv_robust(raw_fp)

    # 2) 列重命名 & 裁剪
    col_map: Dict[str, str] = MIRBASE_CFG["cols"]  # type: ignore
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
def build_mirbase_gene(project_root: Path, *, force: bool = False) -> None:
    """标准化 miRBase表 为 hald_miRBase__std.csv"""
    _build_one(project_root, force=force)

# ════════════════════════════════════════════════════════════════
# 4. CLI 入口
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser(description="标准化 miRbase")
    pa.add_argument("--root", required=True, type=Path, help="HALDxAI-Project 根目录")
    pa.add_argument("--force", action="store_true", help="覆盖已存在 std 文件")
    args = pa.parse_args()

    build_mirbase_gene(args.root, force=args.force)
