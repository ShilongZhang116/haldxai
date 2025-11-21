#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_umls_std.py
=================
一次性把 UMLS RRF（MRCONSO / MRSTY / MRDEF / MRREL）转成 HALD _std 表：

    • hald_umls__node__std.csv
    • hald_umls__rel__std.csv
"""

from __future__ import annotations

import csv
import pathlib
from collections import defaultdict
from typing import Dict, List

import pandas as pd


# ════════════════════════════════════════════════
# 全局常量（TTY 优先级）
# ════════════════════════════════════════════════
TTY_PREF = {"PF": 0, "PN": 1, "PT": 2}  # PF > PN > PT


# ════════════════════════════════════════════════
# 通用读取工具
# ════════════════════════════════════════════════
def read_rrf(path: pathlib.Path, *, usecols, names, chunksize=None):
    """读取 .RRF，去掉末尾空列，pipe 分隔"""
    kwargs = dict(
        sep="|",
        header=None,
        names=names,
        dtype=str,
        quoting=csv.QUOTE_NONE,
        usecols=usecols,
        low_memory=False,
        chunksize=chunksize,
    )
    return pd.read_csv(path, **kwargs)


# ════════════════════════════════════════════════
# 1. 解析 MRCONSO：首选名 / 同义词
# ════════════════════════════════════════════════
def parse_mrconso(
    fp: pathlib.Path,
    *,
    chunk_size: int = 500_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回
    -------
    best_name_df : CUI, entity_name
    syn_df       : CUI, entity_synonyms
    """
    TTY_PREF = {'PF': 0, 'PN': 1, 'PT': 2}
    best_name, synonyms = {}, defaultdict(set)

    def safe(text):
        return text.strip() if isinstance(text, str) else ""

    for chunk in pd.read_csv(
            fp, sep='|', header=None, usecols=[0, 1, 11, 12, 14],
            names=["CUI", "LAT", "SAB", "TTY", "STR"],
            dtype=str, quoting=csv.QUOTE_NONE,
            chunksize=chunk_size, low_memory=False):

        # 1) 选英文首选名
        eng_pref = chunk.query("LAT=='ENG' and TTY in ['PT','PN','PF']")
        for tup in eng_pref.itertuples(index=False):
            rank = TTY_PREF.get(tup.TTY, 99)
            s = safe(tup.STR)
            if s and (tup.CUI not in best_name or rank < best_name[tup.CUI][0]):
                best_name[tup.CUI] = (rank, s)

        # 2) 收集所有英文同义词
        eng = chunk.query("LAT=='ENG'")
        for tup in eng.itertuples(index=False):
            s = safe(tup.STR)
            if s:
                synonyms[tup.CUI].add(s)

    best_df = pd.DataFrame(
        {"CUI": list(best_name), "entity_name": [v[1] for v in best_name.values()]})
    syn_df = pd.DataFrame(
        {"CUI": list(synonyms), "entity_synonyms": [";".join(sorted(v)) for v in synonyms.values()]})

    return best_df, syn_df


# ════════════════════════════════════════════════
# 2. 解析 MRSTY：语义类型
# ════════════════════════════════════════════════
def parse_mrsty(fp: pathlib.Path) -> pd.DataFrame:
    usecols = [0, 1, 3]  # CUI, TUI, STY
    names = ["CUI", "TUI", "semantic_type"]
    df = read_rrf(fp, usecols=usecols, names=names)
    return df


# ════════════════════════════════════════════════
# 3. 解析 MRDEF：定义（优先 MSH > SNOMEDCT_US）
# ════════════════════════════════════════════════
def parse_mrdef(fp: pathlib.Path) -> pd.DataFrame:
    usecols = [0, 4, 5]  # CUI, DEF, SAB
    names = ["CUI", "DEF", "SAB"]
    df = read_rrf(fp, usecols=usecols, names=names)
    df["rank"] = df["SAB"].map({"MSH": 0, "SNOMEDCT_US": 1}).fillna(2)
    df = df.sort_values(["CUI", "rank"]).drop_duplicates("CUI")
    return df[["CUI", "DEF"]].rename(columns={"DEF": "entity_definition"})


# ════════════════════════════════════════════════
# 4. 组装节点表
# ════════════════════════════════════════════════
def _build_nodes(
    conso_best: pd.DataFrame,
    sty_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    def_df: pd.DataFrame,
) -> pd.DataFrame:
    df = conso_best.rename(columns={"CUI": "entity_ref_id"})

    for sub in (sty_df, syn_df, def_df):
        if not sub.empty:
            sub = sub.rename(columns={"CUI": "entity_ref_id"})
            df = df.merge(sub, on="entity_ref_id", how="left")

    df["sab"] = "UMLS"
    df["src_db"] = "umls"
    return df


# ════════════════════════════════════════════════
# 5. 解析 MRREL → 关系表（流式）
# ════════════════════════════════════════════════
def parse_mrrel(
    fp: pathlib.Path,
    *,
    ref2name: dict[str, str],
    chunk_size: int = 300_000,
) -> pd.DataFrame:
    usecols = [0, 3, 4, 7, 10]  # CUI1, REL, CUI2, RELA, SAB
    names = ["CUI1", "REL", "CUI2", "RELA", "SAB"]

    rows: List[dict] = []
    for i, chunk in enumerate(
        read_rrf(fp, usecols=usecols, names=names, chunksize=chunk_size)
    ):
        chunk["rel_type"] = chunk["RELA"].fillna(chunk["REL"])
        tmp = pd.DataFrame(
            {
                "source_entity_ref_id": chunk["CUI1"],
                "source_entity_name": chunk["CUI1"].map(ref2name),
                "target_entity_ref_id": chunk["CUI2"],
                "target_entity_name": chunk["CUI2"].map(ref2name),
                "rel_type": chunk["rel_type"],
                "sab": chunk["SAB"],
            }
        )
        rows.append(tmp)

        if (i + 1) % 20 == 0:
            print(f"    • processed {(i+1)*chunk_size:,} rows of MRREL")

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True).drop_duplicates()
    df["direction"] = "both"
    df["src_db"] = "umls"
    return df


# ════════════════════════════════════════════════
# 6. 主调度
# ════════════════════════════════════════════════
def build_umls(
    project_root: pathlib.Path,
    *,
    meta_subdir: str = r"data/bio_corpus/UMLS/META",
    out_subdir: str = r"data/external_db",
    conso_chunk: int = 500_000,
    rel_chunk: int = 300_000,
    force: bool = False,
):
    meta_dir = project_root / meta_subdir
    std_dir = project_root / out_subdir
    std_dir.mkdir(parents=True, exist_ok=True)

    out_node = std_dir / "hald_umls__node__std.csv"
    out_rel = std_dir / "hald_umls__rel__std.csv"
    if not force and out_node.exists() and out_rel.exists():
        print("⚠️  输出已存在，使用 --force 强制重建")
        return

    # ----------- 逐表解析 -----------
    print("▶ MRCONSO …")
    best_df, syn_df = parse_mrconso(meta_dir / "MRCONSO.RRF", chunk_size=conso_chunk)

    print("▶ MRSTY …")
    sty_df = parse_mrsty(meta_dir / "MRSTY.RRF")

    print("▶ MRDEF …")
    def_df = parse_mrdef(meta_dir / "MRDEF.RRF")

    print("▶ merge → node")
    node_df = _build_nodes(best_df, sty_df, syn_df, def_df)

    # 建立 CUI→name 字典
    ref2name = dict(zip(node_df["entity_ref_id"], node_df["entity_name"]))

    print("▶ MRREL …")
    rel_df = parse_mrrel(
        meta_dir / "MRREL.RRF", ref2name=ref2name, chunk_size=rel_chunk
    )

    # ----------- 写出 -----------
    def _save(df: pd.DataFrame, fp: pathlib.Path):
        df.to_csv(fp, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    _save(node_df, out_node)
    _save(rel_df, out_rel)

    print("✅ UMLS 标准化完成：")
    print(f"   {out_node.name}  ({len(node_df):,} rows)")
    print(f"   {out_rel.name}   ({len(rel_df):,} rows)")


# ════════════════════════════════════════════════
# 7. CLI
# ════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser(description="标准化 UMLS RRF 到 HALD _std")
    pa.add_argument("--root", required=True, type=pathlib.Path, help="项目根目录")
    pa.add_argument("--force", action="store_true", help="覆盖已存在输出")
    pa.add_argument(
        "--meta-subdir",
        default=r"data/bio_corpus/UMLS/META",
        help="META 目录相对 root 的路径",
    )
    pa.add_argument(
        "--out-subdir",
        default=r"data/external_db",
        help="输出 _std 目录相对 root 的路径",
    )
    pa.add_argument("--conso-chunk", type=int, default=500_000, help="MRCONSO 分块")
    pa.add_argument("--rel-chunk", type=int, default=300_000, help="MRREL 分块")
    args = pa.parse_args()

    build_umls(
        args.root,
        meta_subdir=args.meta_subdir,
        out_subdir=args.out_subdir,
        conso_chunk=args.conso_chunk,
        rel_chunk=args.rel_chunk,
        force=args.force,
    )
