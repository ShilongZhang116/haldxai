#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entity_evidence.py
──────────────────
由 LLM 标注实体（src["LlmEnts"]）生成实体-文献证据表 entity_evidence.csv

字段
-----
evidence_pk | pmid | entity_id | entity_name | evidence

用法
-----
from haldxai.enrich.tables.entity_evidence import build_entity_evidence
build_entity_evidence(project_root, src, force=False)
"""
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import nltk

from haldxai.enrich.tables.loader import load_name2id, save_name2id
from haldxai.enrich.tables.utils import alloc_id

_SENT_SPLIT = nltk.tokenize.sent_tokenize


def _first_sentence_contains(entity: str, abstract: str) -> str | None:
    """返回摘要中**首个包含实体名**的句子；若没有则返回 None。"""
    if not isinstance(abstract, str) or not abstract.strip():
        return None

    entity_re = re.compile(re.escape(entity), re.I)
    for sent in _SENT_SPLIT(abstract):
        if entity_re.search(sent):
            return sent.strip()
    return None

# --------------------------------------------------------------------------- #
# 📦 主函数
# --------------------------------------------------------------------------- #
def build_entity_evidence(
    project_root: Path,
    df_articles: pd.DataFrame,
    df_llm_entities: pd.DataFrame,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """构建或更新 *entity_evidence.csv*。

    参数
    ----
    project_root : Path
        HALD 项目根目录（用于存放 database & name2id）
    df_articles : pd.DataFrame
        文章元数据表，需至少包含 `pmid` 与 `abstract` 列。
    df_llm_entities : pd.DataFrame
        LLM 标注实体结果，需包含 `pmid`, `main_text`, `evidence` 列。
    force : bool
        是否强制重建（默认如文件已存在则直接读取）。
    """

    # —— 目录与输出路径 ——
    db_dir = project_root / "data/database"
    db_dir.mkdir(parents=True, exist_ok=True)
    output_csv = db_dir / "entity_evidence.csv"

    if output_csv.exists() and not force:
        print("🟡 entity_evidence.csv 已存在（跳过）。pass `force=True` 以重新生成。")
        return pd.read_csv(output_csv)

    print("▶ 构建 entity_evidence.csv …")

    # —— 0) 映射加载 ——
    name2id = load_name2id(project_root)

    # —— 1) 规范列 ——
    df = (
        df_llm_entities[["pmid", "main_text", "evidence"]]
        .rename(columns={"main_text": "entity_name"})
        .dropna(subset=["pmid", "entity_name"])
        .copy()
    )

    # PMID 统一为字符串（去除科学计数法等异常）
    df["pmid"] = (
        pd.to_numeric(df["pmid"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .replace("0", "")
    )

    df_articles["pmid"] = (
        pd.to_numeric(df_articles["pmid"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .replace("0", "")
    )

    # —— 2) 回填缺失 evidence ——
    # 先构建 pmid → abstract 映射，减少重复字符串匹配成本
    pmid2abs = (
        df_articles[["pmid", "abstract"]]
        .dropna(subset=["pmid", "abstract"])
        .assign(pmid=lambda d: d["pmid"].astype(str))
        .set_index("pmid")["abstract"]
        .to_dict()
    )

    # 定义向量化函数回填
    def _fill_evidence(row):
        if isinstance(row["evidence"], str) and row["evidence"].strip():
            return row["evidence"].strip()

        abs_text = pmid2abs.get(row["pmid"])
        if not abs_text:
            return ""  # 无摘要

        sent = _first_sentence_contains(row["entity_name"], abs_text)
        return sent or ""

    df["evidence"] = df.apply(_fill_evidence, axis=1)

    # —— 3) 生成 entity_id ——
    df["entity_id"] = df["entity_name"].apply(lambda n: alloc_id(name2id, n))

    # —— 4) 添加自增主键 ——
    df.insert(0, "evidence_pk", range(1, len(df) + 1))

    # —— 5) 保存 ——
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✓ entity_evidence 写出 {len(df):,} 行 → {output_csv}")

    # —— 6) 更新 name2id ——
    save_name2id(project_root, name2id)
    print("✓ name2id.json 已更新，当前条数 =", len(name2id))

    return df

