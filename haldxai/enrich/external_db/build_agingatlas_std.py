#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准化 AgingAtlas 数据表
=======================

用法
----
Python / Notebook
    from haldxai.enrich.external_db.agingatlas.build_agingatlas import build_agingatlas
    build_agingatlas(project_root=Path("/abs/path/to/HALDxAI-Project"), force=False)

CLI（统一入口）
    python -m haldxai.enrich.external_db.cli agingatlas \
           --root /abs/path/to/HALDxAI-Project --force
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# ————————————————————————————————————————————————————————————————
# 项目级公共工具（已在 enrich.external_db.io_utils 里实现）
# ————————————————————————————————————————————————————————————————
from haldxai.enrich.external_db.io_utils import read_tsv_robust  # 自动识别分隔符

# ════════════════════════════════════════════════════════════════
# 1. 统一配置：原始→目标文件、列映射
# ════════════════════════════════════════════════════════════════
DATASETS: Dict[str, Dict] = {
    # -------------------------------------------------------------
    # 主目录：data/bio_corpus/AgingAtlas/
    # -------------------------------------------------------------
    "chip_seq_factors": dict(
        raw="data/bio_corpus/AgingAtlas/CHIP-seq_factors.csv",
        out="hald_agingatlas__chip_seq_factors__std.csv",
        cols={
            "Species": "species",
            "Cell Type": "cell_type",
            "Senescence Type": "senescence_type",
            "Factor": "chip_seq_factor",
            "Technology": "chip_seq_technology",
            "GEO": "geo_id",
            "Doi": "article_doi",
            "Publication": "article_title",
        },
    ),
    "aging_related_gene_set_all": dict(
        raw="data/bio_corpus/AgingAtlas/Aging-related gene set_all.csv",
        out="hald_agingatlas__aging_related_gene_set_all__std.csv",
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
            "Gene_ID": "gene_id",
        },
    ),
    "compounds_list_info": dict(
        raw="data/bio_corpus/AgingAtlas/Compounds_List_Info.csv",
        out="hald_agingatlas__compounds_list_info__std.csv",
        cols={
            "compounds": "compound_name",
            "organism": "species",
            "phenotype": "phenotype",
            "pmid": "pubmed_id",
            "rnaseq": "rnaseq",
        },
    ),
    "metabolomics": dict(
        raw="data/bio_corpus/AgingAtlas/Metabolomics_all.csv",
        out="hald_agingatlas__metabolomics__std.csv",
        cols={
            "Biochemical": "biochemical",
            "Species": "species",
            "Cell/Tissue": "cell_or_tissue",
            "Treatment": "treatment",
            "Log2 FC": "log2foldchange",
            "P": "pvalue",
            "P adjust": "pvalue_adjusted",
            "Super Pathway": "super_pathway",
            "Sub Pathway": "sub_pathway",
            "COMP ID": "comp_id",
            "CHEMICAL ID": "chemical_id",
            "PUBCHEM": "pubchem_id",
            "KEGG": "kegg_id",
            "HMDB": "hmdb_id",
        },
    ),
    "senescence_promoting_genes": dict(
        raw="data/bio_corpus/AgingAtlas/Senescence promoting genes based on CRISPR-Cas9.csv",
        out="hald_agingatlas__senescence_promoting_genes__std.csv",
        cols={
            "Positive|rank": "positive_rank",
            "Gene": "gene_symbol",
            "Alias": "gene_alias",
            "Gene Info": "gene_info",
            "Species": "species",
            "KEGG_ID": "kegg_id",
            "KEGG Pathway": "kegg_pathway",
            "Literature_Name": "article_title",
            "Literature_Link": "article_link",
        },
        # 该子表还需要额外拆分 gene_info 字段（见 _extra_postproc）
        extra=True,
    ),
}

# ════════════════════════════════════════════════════════════════
# 2. gene_info 解析工具（仅 senescence_promoting_genes 用）
# ════════════════════════════════════════════════════════════════
_INFO_KV_RE = re.compile(r"\[([^\]:]+):\s*([^\]]*)\]")

def _parse_gene_info(val: str) -> Dict[str, str | float]:
    """把 '[Gene Symbol: KAT7] [Description: xxx]' 拆成列."""
    if pd.isna(val):
        return {}
    m_id = re.match(r"\s*(\d+)", val)          # 前导数字 ⇒ ENTREZ_ID
    entrez = m_id.group(1) if m_id else np.nan

    kv = {k.lower(): v.strip() for k, v in _INFO_KV_RE.findall(val)}
    def safe(k):  # 统一返回，空串→NaN
        v = kv.get(k, "")
        return v if v else np.nan

    return {
        "entrez_id"      : entrez,
        "locus_tag"      : safe("locus tag"),
        "chromosome"     : safe("chromosome"),
        "map_location"   : safe("map location"),
        "gene_description": safe("description"),
        "gene_type"      : safe("gene type"),
        "gene_symbol_parsed": safe("gene symbol").upper() if isinstance(safe("gene symbol"), str) else np.nan,
    }

# ════════════════════════════════════════════════════════════════
# 3. 核心处理函数
# ════════════════════════════════════════════════════════════════
def _process_one(
    raw_fp: Path,
    out_fp: Path,
    col_map: Dict[str, str],
    *,
    need_extra: bool = False,
    force: bool = False,
) -> None:
    """读文件 → 重命名 → 裁剪 → （可选额外处理）→ 写出 csv."""
    if out_fp.exists() and not force:
        print(f"🟡 {out_fp.name} 已存在（跳过，可用 --force 覆盖）")
        return
    if not raw_fp.exists():
        print(f"❌ 缺失源文件：{raw_fp}")
        return

    df = read_tsv_robust(raw_fp)
    df = df.rename(columns=col_map)
    keep = [v for v in col_map.values() if v in df.columns]
    df = df[keep]

    # senescence_promoting_genes 的 gene_info 拆列
    if need_extra and "gene_info" in df.columns:
        info_df = df["gene_info"].apply(_parse_gene_info).apply(pd.Series)

        # 合并 gene_symbol
        if "gene_symbol" in df.columns:
            df["gene_symbol"] = (
                df["gene_symbol"].str.upper().fillna(info_df["gene_symbol_parsed"])
            )
        else:
            df["gene_symbol"] = info_df["gene_symbol_parsed"]

        df = pd.concat(
            [df.drop(columns="gene_info"), info_df.drop(columns="gene_symbol_parsed")],
            axis=1
        )

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(f"✅ {out_fp.name:<60} {len(df):>8,} 行  {df.shape[1]} 列")

# ════════════════════════════════════════════════════════════════
# 4. 对外统一函数
# ════════════════════════════════════════════════════════════════
def build_agingatlas(project_root: Path, *, force: bool = False) -> None:
    """
    标准化 AgingAtlas 全量子表.

    Parameters
    ----------
    project_root : Path
        HALDxAI-Project 根目录
    force : bool
        True ⇒ 覆盖已存在的 std 文件
    """
    std_dir = project_root / "data/external_db"
    for name, cfg in DATASETS.items():
        raw_fp = project_root / cfg["raw"]
        out_fp = std_dir / cfg["out"]
        _process_one(
            raw_fp,
            out_fp,
            cfg["cols"],
            need_extra=cfg.get("extra", False),
            force=force,
        )

# ════════════════════════════════════════════════════════════════
# 5. CLI 入口（供 external_db/cli.py 或单独调用）
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser(description="标准化 AgingAtlas 外部数据库")
    pa.add_argument("--root", required=True, type=Path, help="HALDxAI-Project 根目录")
    pa.add_argument("--force", action="store_true", help="覆盖已存在 std 文件")
    args = pa.parse_args()

    build_agingatlas(args.root, force=args.force)
