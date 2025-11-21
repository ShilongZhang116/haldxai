#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准化 Hall (Human Aging Landmark Library) 外部数据库

Notebook 用法
-------------
from haldxai.enrich.external_db.hall.build_hall_std import build_hall
build_hall(project_root=Path("/abs/path/to/HALDxAI-Project"), force=True)

CLI 用法
--------
$ python -m haldxai.enrich.external_db.cli hall \
        --root /abs/path/to/HALDxAI-Project --force
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Callable

import pandas as pd
from haldxai.enrich.external_db.io_utils import read_tsv_robust  # 统一安全读取

# ════════════════════════════════════════════════════════════════
# 1. 数据集声明
# ════════════════════════════════════════════════════════════════
DATASETS: Dict[str, Dict] = {
    # ------------------------------------------------------------
    # 1) Aging-related Genes（Longevity-map 表）
    # ------------------------------------------------------------
    "longevity_map": dict(
        raw="data/bio_corpus/HALL/Aging-related Genes.csv",
        out="hald_hall__longevity_map__std.csv",
        cols={
            "Symbol": "gene_symbol",
            "Alias": "gene_alias",
            "Description": "gene_description",
            "Function": "gene_function",
            "Gene_Set": "gene_set",
            "Species": "species",
            "Literature_Name": "article_title",
            "Literature_Link": "article_link",
            "KEGG_ID": "kegg_id",
            "KEGG_Name": "kegg_name",
            "Gene_ID": "entrez_id",
        },
    ),
    # ------------------------------------------------------------
    # 2) Genes of Aging-related Diseases
    # ------------------------------------------------------------
    "gene_of_aging_related_diseases": dict(
        raw="data/bio_corpus/HALL/Genes_of_Aging-related_Diseases.csv",
        out="hald_hall__gene_of_aging_related_diseases__std.csv",
        cols={
            "MeshDiseaseCategory": "disease_mesh_category",
            "DiseaseTerm": "disease_name",
            "GeneName": "gene_symbol_list",
            "PMID": "pubmed_id",
        },
    ),
    # ------------------------------------------------------------
    # 3) Longevity-related Genes
    # ------------------------------------------------------------
    "longevity_related_genes": dict(
        raw="data/bio_corpus/HALL/Longevity-related Genes.csv",
        out="hald_hall__longevity_related_genes__std.csv",
        cols={
            "Gene": "gene_symbol",
            "Protein": "protein_name",
            "MainPhysiologicalRole": "main_physiological_role",
            "ChangewithAgeorAbnormility": "change_with_age_or_abnormility",
        },
    ),
    # ------------------------------------------------------------
    # 4) Metabolomics
    # ------------------------------------------------------------
    "metabolomics": dict(
        raw="data/bio_corpus/HALL/Metabolomics.csv",
        out="hald_hall__metabolomics__std.csv",
        cols={
            "Hall ID": "hall_id",
            "Class": "hall_class",
            "Compound Name": "compound_name",
            "Coefficient": "coefficient",
            "P Value": "pvalue",
            "Platform": "platform",
        },
    ),
    # ------------------------------------------------------------
    # 5) Metagenomics
    # ------------------------------------------------------------
    "metagenomics": dict(
        raw="data/bio_corpus/HALL/Metagenomics.csv",
        out="hald_hall__metagenomics__std.csv",
        cols={
            "Hall ID": "hall_id",
            "Genus": "genus",
            "Kindom": "kindom",
            "Class": "class",
            "Age Distributions": "age_distributions",
            "Beta": "beta",
            "P Value": "pvalue",
            "PMID": "pubmed_id",
        },
    ),
    # ------------------------------------------------------------
    # 6) Pharmacogenomics – Quercetin
    # ------------------------------------------------------------
    "pharmacogenomics_quercetin": dict(
        raw="data/bio_corpus/HALL/Pharmacogenomics_quercetin.csv",
        out="hald_hall__pharmacogenomics_quercetin__std.csv",
        cols={
            "Treatment": "treatment",
            "Cell/Tissue": "cell_or_tissue",
            "Differentially Expressed Gene": "differential_expression_gene",
            "P Value": "pvalue",
            "Q Value": "qvalue",
            "log2FC": "log2foldchange",
            "Z Ratio": "z_ration",
            "Doi": "article_doi",
        },
    ),
    # ------------------------------------------------------------
    # 7) Pharmacogenomics – Vitamin C
    # ------------------------------------------------------------
    "pharmacogenomics_vitamin_c": dict(
        raw="data/bio_corpus/HALL/Pharmacogenomics_Vitamin_C.csv",
        out="hald_hall__pharmacogenomics_vitamin_C__std.csv",
        cols={
            "Treatment": "treatment",
            "Cell/Tissue": "cell_or_tissue",
            "Differentially Expressed Gene": "differential_expression_gene",
            "P Value": "pvalue",
            "Q Value": "qvalue",
            "log2FC": "log2foldchange",
            "Z Ratio": "z_ration",
            "Doi": "article_doi",
        },
    ),
    # ------------------------------------------------------------
    # 8) Proteomics
    # ------------------------------------------------------------
    "proteomics": dict(
        raw="data/bio_corpus/HALL/Proteomics.csv",
        out="hald_hall__proteomics__std.csv",
        cols={
            "Hall ID": "hall_id",
            "Cohort Name": "cohort_name",
            "Accession": "accession",
            "Gene Name": "gene_symbol",
            "Cofficient": "coefficient",
            "P Value": "pvalue",
            "P Adjust": "pvalue_adjust",
            "Protein Description": "protein_description",
            "Journal": "journal",
        },
    ),
    # ------------------------------------------------------------
    # 9) Transcriptomics – Age-variant genes
    # ------------------------------------------------------------
    "transcriptomics_age_variant_genes": dict(
        raw="data/bio_corpus/HALL/transcriptomics_age-variant genes.csv",
        out="hald_hall__transcriptomics_age_variant_genes__std.csv",
        cols={
            "HALL_ID": "hall_id",
            "CohortName": "cohort_name",
            "GeneID": "gene_id",
            "Symbol": "gene_symbol",
            "Coefficient": "coefficient",
            "p.value": "pvalue",
            "p.value.adj": "pvalue_adjust",
            "Gene_type": "gene_type",
            "Tissue": "tissue",
        },
    ),
}

# ════════════════════════════════════════════════════════════════
# 2. 通用处理函数
# ════════════════════════════════════════════════════════════════
def _process_one(
    raw_fp: Path,
    out_fp: Path,
    col_map: Dict[str, str],
    loader: Callable[[Path], pd.DataFrame] | None = None,
    force: bool = False,
) -> None:
    """读取 → 重命名 → 裁剪 → 保存 csv."""
    if out_fp.exists() and not force:
        print(f"🟡 {out_fp.name} 已存在，跳过（--force 可覆盖）")
        return
    if not raw_fp.exists():
        print(f"❌ 缺少原始文件：{raw_fp}")
        return

    df = loader(raw_fp) if loader else read_tsv_robust(raw_fp)
    df = df.rename(columns=col_map)
    df = df[[v for v in col_map.values() if v in df.columns]]

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(f"✅ {out_fp.name:<52} {len(df):>8,} 行  {df.shape[1]} 列")


def build_hall(project_root: Path, force: bool = False) -> None:
    """
    批量标准化 Hall 数据库下的所有子表.
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

    pa = argparse.ArgumentParser(description="标准化 Hall 外部数据库")
    pa.add_argument("--root", required=True, type=Path, help="HALDxAI-Project 根目录")
    pa.add_argument("--force", action="store_true", help="覆盖已存在文件")
    args = pa.parse_args()

    build_hall(args.root, force=args.force)
