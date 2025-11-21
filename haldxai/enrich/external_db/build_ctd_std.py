#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准化 CTD 数据表
====================

用法
----
Python / Notebook:
    from haldxai.enrich.external_db.build_ctd_std import build_ctd
    build_ctd(project_root=Path("/abs/path/to/HALDxAI-Project"), force=False)

CLI（统一入口）:
    python -m haldxai.enrich.external_db.cli ctd --root /abs/path/to/HALDxAI-Project --force
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
    "anatomy": dict(
        raw="data/bio_corpus/CTD/CTD_anatomy.csv",
        out="hald_ctd__anatomy__std.csv",
        cols={
            "AnatomyName": "entity_name",
            "AnatomyID": "anatomy_id",
            "Definition": "definition",
            "AltAnatomyIDs": "external_ids",
            "ParentIDs": "patent_ids",
            "TreeNumbers": "tree_numbers",
            "ParentTreeNumbers": "parent_tree_numbers",
            "Synonyms": "synonyms",
            "ExternalSynonyms": "external_synonyms",
        },
    ),
    "chem_gene_ixn_types": dict(
        raw="data/bio_corpus/CTD/CTD_chem_gene_ixn_types.csv",
        out="hald_ctd__chem_gene_ixn_types__std.csv",
        cols={
            "TypeName": "type_name",
            "Code": "type_code",
            "Description": "description",
            "ParentCode": "parent_code",
        },
    ),
    "chem_gene_ixn": dict(
        raw="data/bio_corpus/CTD/CTD_chem_gene_ixns.csv",
        out="hald_ctd__chem_gene_ixn__std.csv",
        cols={
            "ChemicalName": "chemical_name",
            "ChemicalID": "chemical_id",
            "CasRN": "cas_rn",
            "GeneSymbol": "gene_symbol",
            "GeneID": "gene_id",
            "GeneForms": "gene_forms",
            "Organism": "organism_code",
            "Interaction": "interaction_code",
            "InteractionActions": "interaction_actions",
            "InteractionTypes": "interaction_types",
        },
    ),
    "chem_go_enriched": dict(
        raw="data/bio_corpus/CTD/CTD_chem_go_enriched.csv",
        out="hald_ctd__chem_go_enriched__std.csv",
        cols={
            "ChemicalName": "chemical_name",
            "ChemicalID": "chemical_id",
            "CasRN": "cas_rn",
            "Ontology": "ontology",
            "GOTermName": "go_term_name",
            "GOTermID": "go_term_id",
            "HighestGOLevel": "highest_go_level",
            "PValue": "pvalue",
            "CorrectedPValue": "corrected_pvalue",
            "TargetMatchQty": "target_match_qty",
            "TargetTotalQty": "target_total_qty",
            "BackgroundMatchQty": "background_match_qty",
            "BackgroundTotalQty": "background_total_qty",
        },
    ),
    "chem_pathways_enriched": dict(
        raw="data/bio_corpus/CTD/CTD_chem_pathways_enriched.csv",
        out="hald_ctd__chem_pathways_enriched__std.csv",
        cols={
            "ChemicalName": "chemical_name",
            "ChemicalID": "chemical_id",
            "CasRN": "cas_rn",
            "PathwayName": "pathway_name",
            "PathwayID": "pathway_id",
            "PValue": "pvalue",
            "CorrectedPValue": "correlation_pvalue",
            "TargetMatchQty": "target_match_qty",
            "TargetTotalQty": "target_total_qty",
            "BackgroundMatchQty": "background_match_qty",
            "BackgroundTotalQty": "background_total_qty",
        },
    ),
    "chemicals": dict(
        raw="data/bio_corpus/CTD/CTD_chemicals.csv",
        out="hald_ctd__chemicals__std.csv",
        cols={
            "ChemicalName": "chemical_name",
            "ChemicalID": "chemical_id",
            "CasRN": "cas_rn",
            "Definition": "definition",
            "ParentIDs": "parent_ids",
            "TreeNumbers": "tree_numbers",
            "ParentTreeNumbers": "parent_tree_numbers",
            "Synonyms": "synonyms",
        },
    ),
    "chemicals_diseases": dict(
        raw="data/bio_corpus/CTD/CTD_chemicals_diseases.csv",
        out="hald_ctd__chemicals_diseases__std.csv",
        cols={
            "ChemicalName": "chemical_name",
            "ChemicalID": "chemical_id",
            "CasRN": "cas_rn",
            "DiseaseName": "disease_name",
            "DiseaseID": "disease_id",
            "InferenceGeneSymbol": "inference_gene_symbol",
            "InferenceScore": "inference_score",
            "OmimIDs": "omim_ids",
            "PubMedIDs": "pubmed_id",
        },
    ),
    "curated_genes_diseases": dict(
        raw="data/bio_corpus/CTD/CTD_curated_genes_diseases.csv",
        out="hald_ctd__curated_genes_diseases__std.csv",
        cols={
            "GeneSymbol": "gene_symbol",
            "GeneID": "gene_id",
            "DiseaseName": "disease_name",
            "DiseaseID": "disease_id",
            "DirectEvidence": "direct_evidence",
            "OmimIDs": "omim_ids",
            "PubMedIDs": "pubmed_id",
        },
    ),
    "diseases": dict(
        raw="data/bio_corpus/CTD/CTD_diseases.csv",
        out="hald_ctd__diseases__std.csv",
        cols={
            "DiseaseName": "disease_name",
            "DiseaseID": "disease_id",
            "AltDiseaseIDs": "external_disease_ids",
            "Definition": "definition",
            "ParentIDs": "parent_ids",
            "TreeNumbers": "tree_numbers",
            "ParentTreeNumbers": "parent_tree_numbers",
            "Synonyms": "synonyms",
            "SlimMappings": "slim_mappings",
        },
    ),
    "diseases_pathways": dict(
        raw="data/bio_corpus/CTD/CTD_diseases_pathways.csv",
        out="hald_ctd__diseases_pathways__std.csv",
        cols={
            "DiseaseName": "disease_name",
            "DiseaseID": "disease_id",
            "PathwayName": "pathway_name",
            "PathwayID": "pathway_id",
            "InferenceGeneSymbol": "inference_gene_symbol",
        },
    ),
    "scRNA_marker": dict(
        raw="data/bio_corpus/CTD/CTD_exposure_events.csv",
        out="hald_ctd__exposure_events__std.csv",
        cols={
            "exposurestressorname": "exposure_stressor_name",
            "exposurestressorid": "exposure_stressor_id",
            "stressorsourcecategory": "stressors_source_category",
            "stressorsourcedetails": "stressors_source_details",
            "numberofstressorsamples": "number_of_stressor_samples",
            "stressornotes": "stressor_notes",
            "numberofreceptors": "number_of_receptors",
            "receptors": "receptors",
            "receptornotes": "receptor_notes",
            "smokingstatus": "smoking_status",
            "age": "age",
            "ageunitsofmeasurement": "age_units_of_measurement",
            "agequalifier": "age_qual",
            "sex": "sex",
            "race": "race",
            "methods": "methods",
            "detectionlimit": "detection_limit",
            "detectionlimituom": "detection_limit_uom",
            "detectionfrequency": "detection_frequency",
            "medium": "medium",
            "exposuremarker": "exposure_marker",
            "exposuremarkerid": "exposure_marker_id",
            "markerlevel": "marker_level",
            "markerunitsofmeasurement": "marker_units_of_measurement",
            "markermeasurementstatistic": "marker_measurement_statistic",
            "assaynotes": "assay_notes",
            "studycountries": "study_countries",
            "stateorprovince": "state_or_province",
            "citytownregionarea": "city_town_region_area",
            "exposureeventnotes": "exposure_event_notes",
            "outcomerelationship": "outcome_relationship",
            "diseasename": "disease_name",
            "diseaseid": "disease_id",
            "phenotypename": "phenotype_name",
            "phenotypeid": "phenotype_id",
            "phenotypeactiondegreetype": "phenotype_action_degree_type",
            "anatomy": "anatomy",
            "exposureoutcomenotes": "exposure_outcome_notes",
            "reference": "pubmed_id",
            "associatedstudytitles": "associated_study_titles",
            "enrollmentstartyear": "enrollment_start_year",
            "enrollmentendyear": "enrollment_end_year",
            "studyfactors": "study_factors",
        },
    ),
    "exposure_studies": dict(
        raw="data/bio_corpus/CTD/CTD_exposure_studies.csv",
        out="hald_ctd__exposure_studies__std.csv",
        cols={
            "reference": "pubmed_id",
            "studyfactors": "study_factors",
            "exposurestressors": "exposure_stressors",
            "receptors": "receptors",
            "studycountries": "study_countries",
            "mediums": "mediums",
            "exposuremarkers": "exposure_marker",
            "diseases": "diseases",
            "phenotypes": "phenotypes",
            "authorsummary": "authorsummary",
        },
    ),
    "genes": dict(
        raw="data/bio_corpus/CTD/CTD_genes.csv",
        out="hald_ctd__genes__std.csv",
        cols={
            "GeneSymbol": "gene_symbol",
            "GeneName": "gene_name",
            "GeneID": "gene_id",
            "AltGeneIDs": "external_gene_ids",
            "Synonyms": "synonyms",
            "BioGRIDIDs": "biogrid_ids",
            "PharmGKBIDs": "pharmgkb_ids",
            "UniProtIDs": "uniport_ids",
        },
    ),
    "genes_diseases": dict(
        raw="data/bio_corpus/CTD/CTD_genes_diseases.csv",
        out="hald_ctd__genes_diseases__std.csv",
        cols={
            "GeneSymbol": "gene_symbol",
            "GeneID": "gene_id",
            "DiseaseName": "disease_name",
            "DiseaseID": "disease_id",
            "DirectEvidence": "direct_evidence",
            "InferenceChemicalName": "inference_chemical_name",
            "InferenceScore": "inference_score",
            "OmimIDs": "omim_ids",
            "PubMedIDs": "pubmed_id"
        },
    ),
    "genes_pathways": dict(
        raw="data/bio_corpus/CTD/CTD_genes_pathways.csv",
        out="hald_ctd__genes_pathways__std.csv",
        cols={
            "GeneSymbol": "gene_symbol",
            "GeneID": "gene_id",
            "PathwayName": "pathway_name",
            "PathwayID": "pathway_id",
        },
    ),
    "pathways": dict(
        raw="data/bio_corpus/CTD/CTD_pathways.csv",
        out="hald_ctd__pathways__std.csv",
        cols={
            "PathwayName": "pathway_name",
            "PathwayID": "pathway_id",
        },
    ),
    "pheno_term_ixns": dict(
        raw="data/bio_corpus/CTD/CTD_pheno_term_ixns.csv",
        out="hald_ctd__pheno_term_ixns__std.csv",
        cols={
            "chemicalname": "chemical_name",
            "chemicalid": "chemical_id",
            "casrn": "cas_rn",
            "phenotypename": "phenotype_name",
            "phenotypeid": "phenotype_id",
            "comentionedterms": "co_mentioned_terms",
            "organism": "organism",
            "organismid": "organism_id",
            "interaction": "interaction",
            "interactionactions": "interaction_actions",
            "anatomyterms": "anatomy_terms",
            "inferencegenesymbols": "inference_gene_symbols",
            "pubmedids": "pubmed_id"
        },
    ),
    "Phenotype-Disease_biological_process_associations": dict(
        raw="data/bio_corpus/CTD/CTD_Phenotype-Disease_biological_process_associations.csv",
        out="hald_ctd__Phenotype-Disease_biological_process_associations__std.csv",
        cols={
            "GOName": "go_name",
            "GOID": "go_id",
            "DiseaseName": "disease_name",
            "DiseaseID": "disease_id",
            "InferenceChemicalQty": "inference_chemical_qty",
            "InferenceChemicalNames": "inference_chemical_names",
            "InferenceGeneQty": "inference_gene_qty",
            "InferenceGeneSymbols": "inference_gene_symbols",
        },
    ),
    "Phenotype-Disease_cellular_component_associations": dict(
        raw="data/bio_corpus/CTD/CTD_Phenotype-Disease_cellular_component_associations.csv",
        out="hald_ctd__Phenotype-Disease_cellular_component_associations__std.csv",
        cols={
            "GOName": "go_name",
            "GOID": "go_id",
            "DiseaseName": "disease_name",
            "DiseaseID": "disease_id",
            "InferenceChemicalQty": "inference_chemical_qty",
            "InferenceChemicalNames": "inference_chemical_names",
            "InferenceGeneQty": "inference_gene_qty",
            "InferenceGeneSymbols": "inference_gene_symbols",
        },
    ),
    "Phenotype-Disease_molecular_function_associations": dict(
        raw="data/bio_corpus/CTD/CTD_Phenotype-Disease_molecular_function_associations.csv",
        out="hald_ctd__Phenotype-Disease_molecular_function_associations__std.csv",
        cols={
            "GOName": "go_name",
            "GOID": "go_id",
            "DiseaseName": "disease_name",
            "DiseaseID": "disease_id",
            "InferenceChemicalQty": "inference_chemical_qty",
            "InferenceChemicalNames": "inference_chemical_names",
            "InferenceGeneQty": "inference_gene_qty",
            "InferenceGeneSymbols": "inference_gene_symbols",
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

    df = pd.read_csv(raw_fp, low_memory=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    df = df.rename(columns=col_map)
    keep_cols = [v for v in col_map.values() if v in df.columns]
    df = df[keep_cols]

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(f"✅ {out_fp.name:<55}  {len(df):>8,} 行  {df.shape[1]} 列")


def build_ctd(project_root: Path, force: bool = False) -> None:
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

    build_ctd(args.root, force=args.force)
