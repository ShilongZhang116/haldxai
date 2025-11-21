#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准化 AgeAnno 数据表
====================

用法
----
Python / Notebook:
    from haldxai.enrich.external_db.ageanno.build_ageanno import build_ageanno
    build_ageanno(project_root=Path("/abs/path/to/HALDxAI-Project"), force=False)

CLI（统一入口）:
    python -m haldxai.enrich.external_db.cli ageanno --root /abs/path/to/HALDxAI-Project --force
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import pandas as pd

# ────────────────────────────────────────────────────────────────
# 公共工具（项目里已存在的 read_tsv_robust）
# ────────────────────────────────────────────────────────────────
from haldxai.enrich.external_db.io_utils import read_tsv_robust   # 你已有的工具，负责自动识别分隔符

# ════════════════════════════════════════════════════════════════
# 1. 统一配置：原始→目标文件、列映射
# ════════════════════════════════════════════════════════════════
DATASETS: Dict[str, Dict] = {
    # -------- disease_drug --------
    "related_diseases": dict(
        raw="data/bio_corpus/AgeAnno/disease_drug/Related diseases.txt",
        out="hald_ageanno__related_diseases__std.csv",
        cols={
            "geneId": "ageanno_gene_id",
            "geneSymbol": "gene_symbol",
            "DSI": "disease_specificity_index",
            "DPI": "disease_pleiotropy_index",
            "diseaseId": "disease_id",
            "score": "ageanno_score",
            "EI": "evidence_index",
            "YearInitial": "pub_year_initial",
            "YearFinal": "pub_year_final",
            "number of pmid": "number_of_pmid",
            "source": "source",
        },
    ),
    "related_drugs": dict(
        raw="data/bio_corpus/AgeAnno/disease_drug/Related drugs.txt",
        out="hald_ageanno__related_drugs__std.csv",
        cols={
            "ChemicalName": "chemical_name",
            "ChemicalID": "chemical_id",
            "GeneSymbol": "gene_symbol",
            "GeneForms": "gene_form",
            "InteractionActions": "interaction_action",
            "PubMedIDs": "pubmed_id",
        },
    ),
    # -------- scATAC --------
    "aging_related_dar": dict(
        raw="data/bio_corpus/AgeAnno/scATAC/Aging-related DAR.txt",
        out="hald_ageanno__aging_related_differential_accessible_regions__std.csv",
        cols={
            "Tissue": "tissue",
            "cell_type": "cell_type",
            "geneId": "gene_id",
            "SYMBOL": "gene_symbol",
            "Category": "category",
            "change": "change",
            "CHR": "chromosome",
            "start": "chr_start_site",
            "end": "chr_end_site",
            "Log2FC": "log2foldchange",
            "Pval": "pvalue",
            "annotation": "annotation",
        },
    ),
    "co_accessibility": dict(
        raw="data/bio_corpus/AgeAnno/scATAC/coAccessiblity.txt",
        out="hald_ageanno__coAccessiblity__std.csv",
        cols={
            "Tissue": "tissue",
            "Cell_type": "cell_type",
            "Category": "category",
            "Change": "change",
            "Chromosome": "chromosome",
            "Start": "chr_start_site",
            "End": "chr_end_site",
            "Annotation": "annotation",
            "Gene.ID": "gene_id",
            "Symbol": "gene_symbol",
            "Correlated_Start": "correlated_chr_start_site",
            "Correlated_End": "correlated_chr_end_site",
            "Correlated_Annotation": "correlated_annotation",
            "Correlated_Gene.ID": "correlated_gene_id",
            "Correlated_Symbol": "correlated_symbol",
            "Correlation": "correlation",
        },
    ),
    "motif_tf": dict(
        raw="data/bio_corpus/AgeAnno/scATAC/motif-TF.txt",
        out="hald_ageanno__motif_tf__std.csv",
        cols={
            "Tissue": "tissue",
            "cell_type": "cell_type",
            "Category": "category",
            "change": "change",
            "TF": "transcription_factors",
            "Pval": "pvalue",
            "mid_Variability": "mid_population_variability",
            "old_Variability": "old_population_variability",
        },
    ),
    "scATAC_marker": dict(
        raw="data/bio_corpus/AgeAnno/scATAC/scATACmarker.txt",
        out="hald_ageanno__scATAC_marker__std.csv",
        cols={
            "Tissue": "tissue",
            "group_name": "cell_type",
            "seqnames": "chromosome",
            "start": "chr_start_site",
            "end": "chr_end_site",
            "strand": "chr_strand",
            "name": "gene_symbol",
        },
    ),
# -------- scRNA --------
    "aging_related_deg": dict(
        raw="data/bio_corpus/AgeAnno/scRNA/Aging-related DEGs.txt",
        out="hald_ageanno__aging_related_differential_expression_genes__std.csv",
        cols={
            "Tissue": "tissue",
            "group": "category",
            "cell_type": "cell_type",
            "gene": "gene_symbol",
            "p_val": "pvalue",
            "avg_log2FC": "average_log2foldchange",
            "pct.1": "pct_expr_group1",
            "pct.2": "pct_expr_group2",
            "p_val_adj": "pvalue_adjusted",
            "change": "change",
            "isTissuespecific": "is_tissue_specific",
            "isCellTypespecific": "is_cell_type_specific",
        },
    ),
    "cell_cell_communication": dict(
        raw="data/bio_corpus/AgeAnno/scRNA/cell_cell_communication.txt",
        out="hald_ageanno__cell_cell_communication__std.csv",
        cols={
            "Tissue": "tissue",
            "group": "category",
            "interacting_pair": "interacting_pair",
            "partner_a": "partner_a",
            "partner_b": "partner_b",
            "gene_a": "gene_symbol_a",
            "gene_b": "gene_symbol_b",
            "secreted": "secreted",
            "receptor_a": "receptor_a",
            "receptor_b": "receptor_b",
            "annotation_strategy": "annotation_strategy",
            "is_integrin": "is_integrin",
            "cell.type_a": "cell_type_a",
            "cell.type_b": "cell_type_b",
            "exp_means": "expression_means",
            "p.value": "pvalue",
        },
    ),
    "gene_variation_coefficient": dict(
        raw="data/bio_corpus/AgeAnno/scRNA/gene_VariationCoefficient.txt",
        out="hald_ageanno__gene_variation_coefficient__std.csv",
        cols={
            "Tissue": "tissue",
            "group": "category",
            "cell_type": "cell_type_list",
            "Gene": "gene_symbol",
            "Result": "variation_coefficient_result",
        },
    ),
    "pathways": dict(
        raw="data/bio_corpus/AgeAnno/scRNA/Pathways.txt",
        out="hald_ageanno__pathways__std.csv",
        cols={
            "Tissue": "tissue",
            "group": "category",
            "cell_type": "cell_type",
            "UpOrDown": "change",
            "Unnamed: 4": "go_domains",
            "ID": "go_id",
            "Description": "go_description",
            "GeneRatio": "gene_ration",
            "BgRatio": "background_ratio",
            "pvalue": "pvalue",
            "p.adjust": "pvalue_adjusted",
            "qvalue": "qvalue",
            "geneID": "gene_symbol_list",
            "Count": "count",
        },
    ),
    "scRNA_marker": dict(
        raw="data/bio_corpus/AgeAnno/scRNA/scRNAmarker.txt",
        out="hald_ageanno__scRNA_marker__std.csv",
        cols={
            "Tissue": "tissue",
            "Cell type": "cell_type",
            "Marker gene": "gene_symbol",
        },
    ),
    "tf_regulation": dict(
        raw="data/bio_corpus/AgeAnno/scRNA/TF regulon.txt",
        out="hald_ageanno__tf_regulation__std.csv",
        cols={
            "Tissue": "tissue",
            "group": "category",
            "cell_type": "cell_type_list",
            "TF": "transcription_factors",
            "rss_value": "rss_value",
            "Targets_number": "targets_number",
            "image_URL": "image_URL",
            "Targets": "targets_gene_symbol_list",
        },
    ),
}


# ════════════════════════════════════════════════════════════════
# 2. 核心处理函数
# ════════════════════════════════════════════════════════════════
def _process_one(
    raw_fp: Path,
    out_fp: Path,
    col_map: Dict[str, str],
    force: bool = False,
) -> None:
    """读 tsv → 重命名 → 裁剪 → 保存 csv."""
    if out_fp.exists() and not force:
        print(f"🟡 {out_fp.name} 已存在（跳过，可用 --force 覆盖）")
        return
    if not raw_fp.exists():
        print(f"❌ 原始文件缺失：{raw_fp}")
        return

    df = read_tsv_robust(raw_fp)
    df = df.rename(columns=col_map)
    keep_cols = [v for v in col_map.values() if v in df.columns]
    df = df[keep_cols]

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(f"✅ {out_fp.name:<55}  {len(df):>8,} 行  {df.shape[1]} 列")


def build_ageanno(project_root: Path, force: bool = False) -> None:
    """
    标准化 AgeAnno 全部子表.

    Parameters
    ----------
    project_root : Path
        HALDxAI-Project 根目录
    force : bool
        True ⇒ 总是覆盖 std 文件；False ⇒ 若已存在则跳过
    """
    std_dir = project_root / "data/external_db"   # 约定好的统一输出目录
    for name, cfg in DATASETS.items():
        raw_fp = project_root / cfg["raw"]
        out_fp = std_dir / cfg["out"]
        _process_one(raw_fp, out_fp, cfg["cols"], force=force)


# ════════════════════════════════════════════════════════════════
# 3. CLI 入口（被 external_db/cli.py 调用）
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser(description="标准化 AgeAnno 外部数据库")
    pa.add_argument("--root", required=True, type=Path, help="HALDxAI-Project 根目录")
    pa.add_argument("--force", action="store_true", help="覆盖已存在 std 文件")
    args = pa.parse_args()

    build_ageanno(args.root, force=args.force)
