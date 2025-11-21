#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准化 ICD-10-CM 2025（Tabular / Index / Neoplasm / Drug）
保存到：{project_root}/data/standard/external_db/hald_icd10__*.csv
"""
from __future__ import annotations

import csv, json, xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict

import pandas as pd

# ════════════════════════════════════════════════════════════════
# 共用常量
# ════════════════════════════════════════════════════════════════
SRC_DB = "icd10cm"
COL_KIND = {
    "2": "malignant_primary",
    "3": "malignant_secondary",
    "4": "carcinoma_in_situ",
    "5": "benign",
    "6": "uncertain_behavior",
    "7": "unspecified_behavior",
}
INTENT = {
    "2": "accidental",
    "3": "intentional_self_harm",
    "4": "assault",
    "5": "undetermined",
    "6": "adverse_effect",
    "7": "underdosing",
}

# ════════════════════════════════════════════════════════════════
# 1. 四个解析器
# ════════════════════════════════════════════════════════════════
def _clean(txt: str | None) -> bool:
    return bool(txt and txt.strip() and txt.strip() not in {"-", "--"})

def _full_title(elem: ET.Element | None) -> str:
    return "".join(elem.itertext()).strip() if elem is not None else ""

# ---------------- Tabular ----------------
def parse_tabular(xml_fp: Path) -> pd.DataFrame:
    rows, root = [], ET.parse(xml_fp).getroot()

    def all_notes(diag: ET.Element) -> list[str]:
        return [
            n.text.strip()
            for n in diag.findall(".//note")
            if n.text and n.text.strip()
        ]

    def walk(node, chap="", sect=""):
        tag = node.tag.lower()
        if tag == "chapter":
            num = node.findtext("name", "").strip()
            desc = node.findtext("desc", "").strip()
            chap = f"{num} {desc}".strip()
        elif tag == "section":
            sect = node.findtext("desc", "").strip()
        if tag == "diag":
            rows.append(
                {
                    "icd10_code": node.findtext("name", ""),
                    "description": node.findtext("desc", ""),
                    "chapter": chap,
                    "section": sect,
                    "note": json.dumps(all_notes(node), ensure_ascii=False),
                }
            )
        for child in node:
            if child.tag.lower() != "sectionindex":
                walk(child, chap, sect)

    walk(root)
    df = pd.DataFrame(rows).dropna(subset=["icd10_code"])
    df["src_file"], df["src_db"] = xml_fp.name, SRC_DB
    return df


# ---------------- Index (主题词) ----------------
def parse_index(xml_fp: Path) -> pd.DataFrame:
    rows, root = [], ET.parse(xml_fp).getroot()

    def rec(node, stack=None, lvl=0):
        stack = stack or []
        if node.tag in {"mainTerm", "term"}:
            title = node.findtext("title") or node.text
            if title:
                stack.append(title.strip())
        if node.tag == "code":
            rows.append(
                {
                    "index_term": " ".join(stack),
                    "index_level": lvl,
                    "icd10_code": node.text.strip(),
                }
            )
        for ch in node:
            rec(ch, stack.copy(), lvl + (ch.tag == "term"))

    for letter in root.findall(".//letter"):
        rec(letter)
    df = pd.DataFrame(rows).drop_duplicates()
    df["src_file"], df["src_db"] = xml_fp.name, SRC_DB
    return df


# ---------------- Neoplasm ----------------
def parse_neoplasm(xml_fp: Path) -> pd.DataFrame:
    rows, root = [], ET.parse(xml_fp).getroot()

    def rec(node, stack=None):
        stack = stack or []
        if node.tag in {"mainTerm", "term"}:
            title = node.findtext("title")
            if _clean(title):
                stack.append(title.strip())
            lvl = len(stack) - 1
            for cell in node.findall("cell"):
                if _clean(cell.text):
                    rows.append(
                        {
                            "index_term": " ".join(stack),
                            "index_level": lvl,
                            "neoplasm_kind": COL_KIND.get(cell.attrib["col"], ""),
                            "icd10_code": cell.text.strip(),
                        }
                    )
        for ch in node:
            if ch.tag in {"mainTerm", "term"}:
                rec(ch, stack.copy())

    for letter in root.findall(".//letter"):
        rec(letter)
    df = pd.DataFrame(rows).drop_duplicates()
    df["src_file"], df["src_db"] = xml_fp.name, SRC_DB
    return df


# ---------------- Drug / Poison ----------------
def parse_drug(xml_fp: Path) -> pd.DataFrame:
    rows, root = [], ET.parse(xml_fp).getroot()

    def rec(node, lvl=0):
        if node.tag in {"mainTerm", "term"}:
            sub = _full_title(node.find("title"))
            if sub:
                for cell in node.findall("cell"):
                    if _clean(cell.text):
                        rows.append(
                            {
                                "substance": sub,
                                "drug_intent": INTENT.get(cell.attrib["col"], ""),
                                "icd10_code": cell.text.strip(),
                                "index_level": lvl,
                            }
                        )
        for ch in node:
            if ch.tag in {"mainTerm", "term"}:
                rec(ch, lvl + 1)

    for letter in root.findall(".//letter"):
        rec(letter)
    df = pd.DataFrame(rows).drop_duplicates()
    df["src_file"], df["src_db"] = xml_fp.name, SRC_DB
    return df


# ════════════════════════════════════════════════════════════════
# 2. 调度器
# ════════════════════════════════════════════════════════════════
TASKS: Dict[str, Dict[str, str | Callable]] = {
    "tabular": dict(
        raw="icd10cm_tabular_2025.xml",
        out="hald_icd10__tabular__std.csv",
        parser=parse_tabular,
    ),
    "index": dict(
        raw="icd10cm_index_2025.xml",
        out="hald_icd10__index__std.csv",
        parser=parse_index,
    ),
    "neoplasm": dict(
        raw="icd10cm_neoplasm_2025.xml",
        out="hald_icd10__neoplasm__std.csv",
        parser=parse_neoplasm,
    ),
    "drug": dict(
        raw="icd10cm_drug_2025.xml",
        out="hald_icd10__drug__std.csv",
        parser=parse_drug,
    ),
}


def _run_one(
    raw_fp: Path,
    out_fp: Path,
    parser: Callable[[Path], pd.DataFrame],
    force: bool = False,
):
    if out_fp.exists() and not force:
        print(f"🟡 {out_fp.name} 已存在，跳过（--force 可覆盖）")
        return
    if not raw_fp.exists():
        print(f"❌ 缺少原始文件：{raw_fp}")
        return
    df = parser(raw_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(f"✅ {out_fp.name:<35} {len(df):>8,} 行  {df.shape[1]} 列")


def build_icd10(project_root: Path, force: bool = False):
    """
    解析 ICD-10-CM 2025 四张 XML 并写出标准表
    """
    raw_dir = project_root / "data/bio_corpus/ICD-10"
    std_dir = project_root / "data/external_db"
    for key, cfg in TASKS.items():
        raw_fp = raw_dir / cfg["raw"]
        out_fp = std_dir / cfg["out"]
        _run_one(raw_fp, out_fp, cfg["parser"], force=force)


# ════════════════════════════════════════════════════════════════
# 3. CLI
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser(description="标准化 ICD-10-CM 2025")
    pa.add_argument("--root", required=True, type=Path, help="HALDxAI-Project 根目录")
    pa.add_argument("--force", action="store_true", help="覆盖已存在文件")
    args = pa.parse_args()

    build_icd10(args.root, force=args.force)
