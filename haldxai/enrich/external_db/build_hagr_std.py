#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准化 HAGR 系列数据表
====================

Python / Notebook
-----------------
>>> from haldxai.enrich.external_db.hagr.build_hagr_std import build_hagr
>>> build_hagr(project_root=Path("/abs/path/to/HALDxAI-Project"), force=False)

CLI（由 external_db/cli.py 统一转发）
------------------------------------
$ python -m haldxai.enrich.external_db.cli hagr --root /abs/path/to/HALDxAI-Project --force
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Callable

import pandas as pd

# 项目内已有的安全读文件工具（能自动识别制表符 / 逗号分隔、混合编码等）
from haldxai.enrich.external_db.io_utils import read_tsv_robust

# ════════════════════════════════════════════════════════════════
# 1. 数据集统一配置
# ════════════════════════════════════════════════════════════════
def _csv_semicolon(fp: Path) -> pd.DataFrame:
    """CellSignatures 的特殊分号 csv（GBK 编码）"""
    return pd.read_csv(fp, sep=";", encoding="gbk")

DATASETS: Dict[str, Dict] = {
    # ---------------------------------------------------------------------
    # CellAge
    # ---------------------------------------------------------------------
    "cell_age": dict(
        raw="data/bio_corpus/HAGR/CellAge/cellAge/cellAge_cellage.tsv",
        out="hald_hagr__cell_age__std.csv",
        cols={
            "Entrez ID": "entrez_id",
            "Gene symbol": "gene_symbol",
            "Gene name": "gene_description",
            "Cancer Cell": "is_cancer_cell",
            "Type of senescence": "type_of_senescence",
            "Senescence Effect": "senescence_effect",
            "Reference": "pubmed_id",
        },
    ),
    "cell_age_senescence_genes": dict(
        raw="data/bio_corpus/HAGR/CellAge/CellAge Senescence Genes.csv",
        out="hald_hagr__cell_age_senescence_genes__std.csv",
        cols={
            "Entrez Id": "entrez_id",
            "Gene Symbol": "gene_symbol",
            "Method": "method",
            "Cell Types": "cell_type",
            "Cell Lines": "cell_line",
            "Cancer Line?": "is_cell_line",
            "Senescence Type": "senescence_type",
            "Senescence Effect": "senescence_effect",
        },
    ),

    # ---------------------------------------------------------------------
    # Cell signatures（分号分隔，需要自定义加载器）
    # ---------------------------------------------------------------------
    "cell_signatures": dict(
        raw="data/bio_corpus/HAGR/CellAge/cellSignatures/signatures.csv",
        out="hald_hagr__cell_signatures__std.csv",
        cols={
            "gene_symbol": "gene_symbol",
            "gene_name": "gene_description",
            "entrez_id": "entrez_id",
            "total": "total",
            "ovevrexp": "ovevrexpression",
            "underexp": "underexpression",
            "p_value": "pvalue",
        },
        loader=_csv_semicolon,  # 特殊读取函数
    ),

    # ---------------------------------------------------------------------
    # DrugAge
    # ---------------------------------------------------------------------
    "drug_age": dict(
        raw="data/bio_corpus/HAGR/DugAge/drugage.csv",
        out="hald_hagr__drug_age__std.csv",
        cols={
            "compound_name": "compound",
            "species": "species",
            "strain": "strain",
            "dosage": "dosage",
            "age_at_initiation": "age_at_initiation",
            "treatment_duration": "treatment_duration",
            "avg_lifespan_change_percent": "avg_lifespan_change_percent",
            "avg_lifespan_significance": "avg_lifespan_significance",
            "max_lifespan_change_percent": "max_lifespan_change_percent",
            "max_lifespan_significance": "max_lifespan_significance",
            "gender": "gender",
            "weight_change_percent": "weight_change_percent",
            "weight_change_significance": "weight_change_significance",
            "ITP": "ITP",
            "pubmed_id": "pubmed_id",
        },
    ),

    # ---------------------------------------------------------------------
    # GenAge (human)
    # ---------------------------------------------------------------------
    "genage_human": dict(
        raw="data/bio_corpus/HAGR/GenAge/genage_human.csv",
        out="hald_hagr__genage_human__std.csv",
        cols={
            "GenAge ID": "genage_id",
            "symbol": "gene_symbol",
            "name": "gene_description",
            "entrez gene id": "entrez_id",
            "uniprot": "uniport",
            "why": "source",
        },
    ),

    # ---------------------------------------------------------------------
    # LongevityMap
    # ---------------------------------------------------------------------
    "longevity_map": dict(
        raw="data/bio_corpus/HAGR/LongevityMap/longevity.csv",
        out="hald_hagr__longevity_map__std.csv",
        cols={
            "id": "longevity_map_id",
            "Association": "association",
            "Population": "population",
            "Variant(s)": "variant",
            "Gene(s)": "gene_symbol",
            "PubMed": "pubmed_id",
        },
    ),
}

# ════════════════════════════════════════════════════════════════
# 2. 公共处理器
# ════════════════════════════════════════════════════════════════
def _process_one(
    raw_fp: Path,
    out_fp: Path,
    col_map: Dict[str, str],
    loader: Callable[[Path], pd.DataFrame] | None = None,
    force: bool = False,
) -> None:
    """读取 → 重命名列 → 裁剪 → 输出."""
    if out_fp.exists() and not force:
        print(f"🟡 {out_fp.name} 已存在（跳过，可 --force 覆盖）")
        return
    if not raw_fp.exists():
        print(f"❌ 缺少原始文件：{raw_fp}")
        return

    # 选择加载方式：默认 read_tsv_robust；某些表提供自定义 loader
    df = loader(raw_fp) if loader else read_tsv_robust(raw_fp)

    df = df.rename(columns=col_map)
    df = df[[v for v in col_map.values() if v in df.columns]]

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(f"✅ {out_fp.name:<45} {len(df):>8,} 行  {df.shape[1]} 列")


def build_hagr(project_root: Path, force: bool = False) -> None:
    """
    批量生成 HAGR 各子表的标准化 csv.

    Parameters
    ----------
    project_root : Path
        HALDxAI-Project 根目录
    force : bool
        True → 覆盖已有；False → 已存在时跳过
    """
    std_dir = project_root / "data/external_db"
    for name, cfg in DATASETS.items():
        raw_fp = project_root / cfg["raw"]
        out_fp = std_dir / cfg["out"]
        _process_one(
            raw_fp,
            out_fp,
            cfg["cols"],
            loader=cfg.get("loader"),
            force=force,
        )

# ════════════════════════════════════════════════════════════════
# 3. CLI 入口
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser(description="标准化 HAGR 外部数据库")
    pa.add_argument("--root", required=True, type=Path, help="HALDxAI-Project 根目录")
    pa.add_argument("--force", action="store_true", help="覆盖已存在 std 文件")
    args = pa.parse_args()

    build_hagr(args.root, force=args.force)
