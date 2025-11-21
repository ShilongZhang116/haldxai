#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准化 UniProt Swiss-Prot XML ➜ CSV
-------------------------------------------------
输出：hald_uniport__uniport__std.csv
"""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


# ════════════════════════════════════════════════
# 1. 单条记录解析器
# ════════════════════════════════════════════════
_U_NS = "{https://uniprot.org/uniprot}"   # UniProt XML 命名空间前缀


def _tx(node: ET.Element | None, default: str = "") -> str:
    """安全取文本 + strip"""
    return (node.text or default).strip() if node is not None else default


def parse_uniprot_xml(xml_fp: Path) -> pd.DataFrame:
    """
    流式解析 UniProt (Swiss-Prot) XML，生成标准列 DataFrame
    """

    recs: list[dict] = []

    # 仅在 <entry> 结束事件时解析，节省内存
    for evt, entry in ET.iterparse(xml_fp, events=("end",)):
        if entry.tag != _U_NS + "entry":
            continue

        # accession（取第一个）
        acc = _tx(entry.find(f"{_U_NS}accession"))

        # 名称相关
        full_name = _tx(
            entry.find(
                f"{_U_NS}protein/{_U_NS}recommendedName/{_U_NS}fullName"
            )
        )
        rec_short = _tx(
            entry.find(
                f"{_U_NS}protein/{_U_NS}recommendedName/{_U_NS}shortName"
            )
        )
        alt_short = _tx(
            entry.find(
                f"{_U_NS}protein/{_U_NS}alternativeName/{_U_NS}shortName"
            )
        )

        gene_name = _tx(entry.find(f"{_U_NS}gene/{_U_NS}name[@type='primary']"))

        organism = _tx(
            entry.find(f"{_U_NS}organism/{_U_NS}name[@type='scientific']")
        )

        hosts = ", ".join(
            _tx(h)
            for h in entry.findall(
                f"{_U_NS}organismHost/{_U_NS}name[@type='scientific']"
            )
        )

        function = _tx(
            entry.find(f"{_U_NS}comment[@type='function']/{_U_NS}text")
        )

        seq_elem = entry.find(f"{_U_NS}sequence")
        sequence = _tx(seq_elem)

        # 第一个 dbReference 作示例（可根据需要扩展）
        db_ref = entry.find(f"{_U_NS}dbReference")
        db_id = db_ref.attrib.get("id", "") if db_ref is not None else ""

        recs.append(
            dict(
                uniprot_id=acc,
                full_name=full_name,
                recommended_name=rec_short,
                alt_short_name=alt_short,
                gene_name=gene_name,
                organism=organism,
                hosts=hosts,
                function=function,
                db_reference=db_id,
                sequence=sequence,
            )
        )

        # 及时 clear，释放内存
        entry.clear()

    return pd.DataFrame.from_records(recs)


# ════════════════════════════════════════════════
# 2. 调度函数
# ════════════════════════════════════════════════
def build_uniprot(
    project_root: Path,
    raw_name: str = "uniprot_sprot.xml",
    force: bool = False,
):
    """
    解析 UniProt XML 并写标准化 CSV

    Parameters
    ----------
    project_root : Path
        HALDxAI 项目根目录
    raw_name : str
        原始 XML 文件名（默认 `uniprot_sprot.xml`）
    force : bool
        True 则覆盖已存在的输出
    """
    raw_fp = project_root / "data/bio_corpus/Uniport" / raw_name
    std_dir = project_root / "data/external_db"
    std_dir.mkdir(parents=True, exist_ok=True)
    out_fp = std_dir / "hald_uniport__uniport__std.csv"

    if out_fp.exists() and not force:
        print(f"⚠️  {out_fp.name} 已存在，跳过；如需覆盖请加 --force")
        return

    print(f"▶ 开始解析 {raw_fp.name} …")
    df = parse_uniprot_xml(raw_fp)

    df.to_csv(
        out_fp, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL
    )
    print(f"✅ 标准化完成 → {out_fp}  ({len(df):,} 行, {df.shape[1]} 列)")


# ════════════════════════════════════════════════
# 3. CLI
# ════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser(description="标准化 UniProt Swiss-Prot XML")
    pa.add_argument(
        "--root",
        required=True,
        type=Path,
        help="HALDxAI-Project 根目录（绝对路径）",
    )
    pa.add_argument(
        "--raw-name",
        default="uniprot_sprot.xml",
        help="原始 UniProt XML 文件名",
    )
    pa.add_argument("--force", action="store_true", help="已存在时覆盖写出")
    args = pa.parse_args()

    build_uniprot(args.root, raw_name=args.raw_name, force=args.force)
