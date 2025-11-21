#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准化 OpenGenes 数据表
====================

用法
----
Python / Notebook:
    from haldxai.enrich.external_db.build_opengenes_std import build_opengenes
    build_opengenes(project_root=Path("/abs/path/to/HALDxAI-Project"), force=False)

CLI（统一入口）:
    python -m haldxai.enrich.external_db.cli opengenes --root /abs/path/to/HALDxAI-Project --force
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
    "longevity_gene": dict(
        raw="data/bio_corpus/OpenGenes/article data/Genes associated with human longevity and increased mammalian lifespan.csv",
        out="hald_opengenes__longevity_gene__std.csv",
        cols={
            "Gene symbol": "gene_symbol",
            "Gene name": "gene_description",
            "Organism, ortholog and number of studies/entries* for data on increased lifespan": "species_orthologs",
            "Effect on gene function, increased lifespan": "effect_on_gene_function",
            "Longevity polymorphisms": "longevity_polymorphisms",
        },
    ),
    "gene_aging_mechanisms": dict(
        raw="data/bio_corpus/OpenGenes/general data/gene-aging-mechanisms.tsv",
        out="hald_opengenes__gene_aging_mechanisms__std.csv",
        cols={
            "hgnc": "gene_symbol",
            "hallmarks_of_aging": "hallmarks_of_aging",
        },
    ),
    "gene_confidence_level": dict(
        raw="data/bio_corpus/OpenGenes/general data/gene-confidence-level.tsv",
        out="hald_opengenes__gene_confidence_level__std.csv",
        cols={
            "hgnc": "gene_symbol",
            "confidence_level": "confidence_level",
        },
    ),
    "gene_criteria": dict(
        raw="data/bio_corpus/OpenGenes/general data/gene-criteria.tsv",
        out="hald_opengenes__gene_criteria__std.csv",
        cols={
            "hgnc": "gene_symbol",
            "criteria": "criteria",
        },
    ),
    "gene_disease": dict(
        raw="data/bio_corpus/OpenGenes/general data/gene-diseases.tsv",
        out="hald_opengenes__gene_disease__std.csv",
        cols={
            "hgnc": "gene_symbol",
            "diseases": "diseases",
        },
    ),
    "gene_evolution": dict(
        raw="data/bio_corpus/OpenGenes/general data/gene-evolution.tsv",
        out="hald_opengenes__gene_evolution__std.csv",
        cols={
            "hgnc": "gene_symbol",
            "gene_origin": "gene_origin",
            "gene_family_origin": "gene_family_origin",
            "conservative_in": "conservative_in",
        },
    ),
    "age_related_changes": dict(
        raw="data/bio_corpus/OpenGenes/research data/age-related-changes.tsv",
        out="hald_opengenes__age_related_changes__std.csv",
        cols={
            "hgnc": "gene_symbol",
            "model_organism": "model_organism",
            "line": "line",
            "sex": "gender",
            "change_percentage": "change_percentage",
            "p_value": "pvalue",
            "sample": "sample",
            "age_of_control_min": "age_of_control_min",
            "age_of_control_mean": "age_of_control_mean",
            "age_of_control_max": "age_of_control_max",
            "age_of_experiment_min": "age_of_experiment_min",
            "age_of_experiment_mean": "age_of_experiment_mean",
            "age_of_experiment_max": "age_of_experiment_max",
            "age_unit": "age_unit",
            "change_type": "change_type",
            "control_cohort_size": "control_cohort_size",
            "experiment_cohort_size": "experiment_cohort_size",
            "statistical_method": "statistical_method",
            "expression_evaluation_by": "expression_evaluation_by",
            "measurement_method": "measurement_method",
            "comment": "comment",
            "doi": "article_doi",
            "pmid": "pubmed_id",
        },
    ),
    "age_related_processes": dict(
        raw="data/bio_corpus/OpenGenes/research data/age-related-processes-change.tsv",
        out="hald_opengenes__age_related_processes__std.csv",
        cols={
            "hgnc": "gene_symbol",
            "comment": "comment",
            "doi": "article_doi",
            "pmid": "pubmed_id",
            "intervention": "intervention",
            "model_organism": "model_organism",
            "line": "line",
            "intervention_deteriorates": "intervention_deteriorates",
            "intervention_improves": "intervention_improves",
            "intervention_result": "intervention_result",
            "process": "process",
            "age": "age",
            "genotype": "genotype",
            "sex": "gender",
        },
    ),
    "association_with_longevity": dict(
        raw="data/bio_corpus/OpenGenes/research data/associations-with-longevity.tsv",
        out="hald_opengenes__association_with_longevity__std.csv",
        cols={
            "hgnc": "gene_symbol",
            "polymorphism_type": "polymorphism_type",
            "polymorphism_id": "polymorphism_id",
            "nucleotide_substitution": "nucleotide_substitution",
            "amino_acid_substitution": "amino_acid_substitution",
            "polymorphism_other": "polymorphism_other",
            "effect": "effect",
            "association_type": "association_type",
            "significance": "significance",
            "p_value": "pvalue",
            "change_type": "change_type",
            "control_cohort_size": "control_cohort_size",
            "experiment_cohort_size": "experiment_cohort_size",
            "control_lifespan_min": "control_lifespan_min",
            "control_lifespan_mean": "control_lifespan_mean",
            "control_lifespan_max": "control_lifespan_max",
            "experiment_lifespan_min": "experiment_lifespan_min",
            "experiment_lifespan_mean": "experiment_lifespan_mean",
            "experiment_lifespan_max": "experiment_lifespan_max",
            "ethnicity": "ethnicity",
            "associated_allele": "associated_allele",
            "non-associated_allele": "non-associated_allele",
            "allelic_frequency_controls": "allelic_frequency_controls",
            "allelic_frequency_experiment": "allelic_frequency_experiment",
            "study_type": "study_type",
            "sex": "gender",
            "doi": "article_doi",
            "pmid": "pubmed_id",
            "comment": "comment",
        },
    ),
    "gene_regulation": dict(
        raw="data/bio_corpus/OpenGenes/research data/gene-regulation.tsv",
        out="hald_opengenes__gene_regulation__std.csv",
        cols={
            "hgnc": "gene_symbol",
            "comment": "comment",
            "doi": "article_doi",
            "pmid": "pubmed_id",
            "protein_activity": "protein_activity",
            "regulated_gene": "regulated_gene",
            "regulation_type": "regulation_type",
        },
    ),
    "lifespan_change": dict(
        raw="data/bio_corpus/OpenGenes/research data/lifespan-change.tsv",
        out="hald_opengenes__lifespan_change__std.csv",
        cols={
            "hgnc": "gene_symbol",
            "model_organism": "model_organism",
            "sex": "gender",
            "line": "line",
            "effect_on_lifespan": "effect_on_lifespan",
            "control_cohort_size": "control_cohort_size",
            "experiment_cohort_size": "experiment_cohort_size",
            "quantity_of_animals_in_cage_or_container": "quantity_of_animals_in_cage_or_container",
            "containment_t_celsius_from": "containment_t_celsius_from",
            "containment_t_celsius_to": "containment_t_celsius_to",
            "diet": "diet",
            "target_gene_expression_change": "target_gene_expression_change",
            "control_lifespan_min": "control_lifespan_min",
            "control_lifespan_mean": "control_lifespan_mean",
            "control_lifespan_median": "control_lifespan_median",
            "control_lifespan_max": "control_lifespan_max",
            "experiment_lifespan_min": "experiment_lifespan_min",
            "experiment_lifespan_mean": "experiment_lifespan_mean",
            "experiment_lifespan_median": "experiment_lifespan_median",
            "experiment_lifespan_max": "experiment_lifespan_max",
            "lifespan_time_unit": "lifespan_time_unit",
            "lifespan_%_change_min": "lifespan_%_change_min",
            "significance_min": "significance_min",
            "lifespan_%_change_mean": "lifespan_%_change_mean",
            "significance_mean": "significance_mean",
            "lifespan_%_change_median": "lifespan_%_change_median",
            "significance_median": "significance_median",
            "lifespan_%_change_max": "lifespan_%_change_max",
            "significance_max": "significance_max",
            "intervention_deteriorates": "intervention_deteriorates",
            "intervention_improves": "intervention_improves",
            "main_effect_on_lifespan": "main_effect_on_lifespan",
            "intervention_way": "intervention_way",
            "intervention_method": "intervention_method",
            "genotype": "genotype",
            "tissue": "tissue",
            "promoter_or_driver": "promoter_or_driver",
            "induction_by_drug_withdrawal": "induction_by_drug_withdrawal",
            "drug": "drug",
            "treatment_start": "treatment_start",
            "treatment_end": "treatment_end",
            "doi": "article_doi",
            "pmid": "pubmed_id",
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

    try:
        df = read_tsv_robust(raw_fp)
    except UnicodeDecodeError:
        df = pd.read_csv(raw_fp, encoding='utf-8-sig', sep='\t', low_memory=False)

    df = df.rename(columns=col_map)
    keep_cols = [v for v in col_map.values() if v in df.columns]
    df = df[keep_cols]

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(f"✅ {out_fp.name:<55}  {len(df):>8,} 行  {df.shape[1]} 列")


def build_opengenes(project_root: Path, force: bool = False) -> None:
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

    pa = argparse.ArgumentParser(description="标准化 OpenGenes 外部数据库")
    pa.add_argument("--root", required=True, type=Path, help="HALDxAI-Project 根目录")
    pa.add_argument("--force", action="store_true", help="覆盖已存在 std 文件")
    args = pa.parse_args()

    build_opengenes(args.root, force=args.force)
