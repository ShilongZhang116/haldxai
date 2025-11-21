#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准化 MeSH baseline（Desc / Qual / Supp / PA 4 个主表 + 2 个关系表）
--------------------------------------------------------------------
生成文件
└─ {project_root}/data/standard/external_db/
   ├─ hald_mesh__desc__std.csv
   ├─ hald_mesh__qual__std.csv
   ├─ hald_mesh__supp__std.csv
   ├─ hald_mesh__pa__std.csv
   ├─ hald_mesh__concept_rel__std.csv
   ├─ hald_mesh__pharm_action_rel__std.csv
   └─ hald_mesh__pa_descriptor2substance.csv   (可选)
"""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, Tuple

import pandas as pd

SRC_DB = "mesh"


# ════════════════════════════════════════════════
# 共用工具
# ════════════════════════════════════════════════
def _tx(el: ET.Element | None, default: str = "") -> str:
    """安全取文本并 `.strip()`"""
    return (el.text or default).strip() if el is not None else default


def _collect_terms(concept_list: ET.Element | None, preferred: str) -> set[str]:
    """收集 ConceptList 中的全部同义词（去重 + 去首位空）"""
    syns: set[str] = set()
    if concept_list is None:
        return syns
    for c in concept_list.findall("Concept"):
        for t in c.findall("Term"):
            s = _tx(t.find("String"))
            if s and s.lower() != preferred.lower():
                syns.add(s)
    return syns


def _write(df: pd.DataFrame, out_fp: Path):
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        out_fp,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )
    print(f"✅ {out_fp.name:<40}  {len(df):>9,} 行  {df.shape[1]} 列")


# ════════════════════════════════════════════════
# 1. 单文件解析器
# ════════════════════════════════════════════════
def parse_descriptor(xml_fp: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Desc 主表 + ConceptRelation + PA link"""
    desc_rows, crel_rows, pa_rows = [], [], []

    for _evt, rec in ET.iterparse(xml_fp, events=("end",)):
        if rec.tag != "DescriptorRecord":
            continue

        ui = _tx(rec.find("DescriptorUI"))
        name = _tx(rec.find("DescriptorName/String"))

        desc_rows.append(
            dict(mesh_id=ui, entity_name=name, category="desc", src_file=xml_fp.name, src_db=SRC_DB)
        )

        # --- ConceptRelationList
        for concept in rec.findall(".//Concept"):
            crl = concept.find("ConceptRelationList")
            if crl is None:
                continue
            for rel in crl.findall("ConceptRelation"):
                crel_rows.append(
                    dict(
                        descriptor_ui=ui,
                        concept1_ui=_tx(rel.find("Concept1UI")),
                        concept2_ui=_tx(rel.find("Concept2UI")),
                        relation_name=rel.attrib.get("RelationName", ""),
                    )
                )

        # --- PharmacologicalAction link
        for pa in rec.findall(".//PharmacologicalActionList/PharmacologicalAction"):
            pa_rows.append(
                dict(
                    descriptor_ui=ui,
                    pa_ui=_tx(pa.find("DescriptorReferredTo/DescriptorUI")),
                    pa_name=_tx(pa.find("DescriptorReferredTo/DescriptorName/String")),
                )
            )

        rec.clear()  # 释放内存

    return (
        pd.DataFrame(desc_rows),
        pd.DataFrame(crel_rows).drop_duplicates(),
        pd.DataFrame(pa_rows).drop_duplicates(),
    )


def parse_qualifier(xml_fp: Path) -> pd.DataFrame:
    rows = []
    for _evt, rec in ET.iterparse(xml_fp, events=("end",)):
        if rec.tag != "QualifierRecord":
            continue
        ui = _tx(rec.find("QualifierUI"))
        name = _tx(rec.find("QualifierName/String"))
        rows.append(dict(mesh_id=ui, entity_name=name, category="qual", src_file=xml_fp.name, src_db=SRC_DB))
        rec.clear()
    return pd.DataFrame(rows)


def parse_supplement(xml_fp: Path) -> pd.DataFrame:
    rows = []
    for _evt, rec in ET.iterparse(xml_fp, events=("end",)):
        if rec.tag != "SupplementalRecord":
            continue
        ui = _tx(rec.find("SupplementalRecordUI"))
        name = _tx(rec.find("SupplementalRecordName/String"))
        rows.append(dict(mesh_id=ui, entity_name=name, category="supp", src_file=xml_fp.name, src_db=SRC_DB))
        rec.clear()
    return pd.DataFrame(rows)


def parse_pa(xml_fp: Path, want_xref: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
    pa_rows, xref_rows = [], []
    for _evt, elem in ET.iterparse(xml_fp, events=("end",)):
        if elem.tag != "PharmacologicalAction":
            continue
        ref = elem.find("DescriptorReferredTo")
        ui = _tx(ref.find("DescriptorUI"))
        nm = _tx(ref.find("DescriptorName/String"))
        pa_rows.append(dict(mesh_id=ui, entity_name=nm, category="pa", src_file=xml_fp.name, src_db=SRC_DB))

        if want_xref:
            for sub in elem.findall(".//PharmacologicalActionSubstanceList/Substance"):
                xref_rows.append(
                    dict(
                        descriptor_ui=ui,
                        substance_ui=_tx(sub.find("RecordUI")),
                        substance_name=_tx(sub.find("RecordName/String")),
                    )
                )
        elem.clear()
    pa_df = pd.DataFrame(pa_rows).drop_duplicates()
    xref_df = pd.DataFrame(xref_rows).drop_duplicates() if want_xref else None
    return pa_df, xref_df


# ════════════════════════════════════════════════
# 2. 调度器
# ════════════════════════════════════════════════
def build_mesh(project_root: Path, year: int = 2025, force: bool = False, want_xref: bool = True):
    """
    解析 MeSH baseline 若干 XML，写出 4 张主表+关系表
    """
    raw_dir = project_root / "data/bio_corpus/MeSH"
    std_dir = project_root / "data/external_db"
    std_dir.mkdir(parents=True, exist_ok=True)

    xml_desc = raw_dir / f"desc{year}.xml"
    xml_qual = raw_dir / f"qual{year}.xml"
    xml_supp = raw_dir / f"supp{year}.xml"
    xml_pa = raw_dir / f"pa{year}.xml"

    # 1️⃣ Descriptor
    desc_df, crel_df, pa_rel_df = parse_descriptor(xml_desc)
    _write(desc_df, std_dir / "hald_mesh__desc__std.csv")
    _write(crel_df, std_dir / "hald_mesh__concept_rel__std.csv")
    _write(pa_rel_df, std_dir / "hald_mesh__pharm_action_rel__std.csv")

    # 2️⃣ Qualifier
    qual_df = parse_qualifier(xml_qual)
    _write(qual_df, std_dir / "hald_mesh__qual__std.csv")

    # 3️⃣ Supplemental
    supp_df = parse_supplement(xml_supp)
    _write(supp_df, std_dir / "hald_mesh__supp__std.csv")

    # 4️⃣ Pharmacological Action
    pa_df, xref_df = parse_pa(xml_pa, want_xref=want_xref)
    _write(pa_df, std_dir / "hald_mesh__pa__std.csv")
    if want_xref and xref_df is not None and not xref_df.empty:
        _write(xref_df, std_dir / "hald_mesh__pa_descriptor2substance.csv")

    print("\n🎉 MeSH 标准化完成！")


# ════════════════════════════════════════════════
# 3. CLI
# ════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser(description="标准化 MeSH baseline (desc/qual/supp/pa)")
    pa.add_argument("--root", required=True, type=Path, help="HALDxAI-Project 根目录")
    pa.add_argument("--year", default=2025, type=int, help="baseline 年份 (默认 2025)")
    pa.add_argument("--no-xref", action="store_true", help="不生成 PA↔Substance 对照表")
    args = pa.parse_args()

    build_mesh(args.root, year=args.year, want_xref=not args.no_xref)
