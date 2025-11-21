# haldxai/enrich/tables/relation_evidence.py
from __future__ import annotations
import json, re, hashlib, logging
from pathlib import Path
from typing import Optional, Dict, Tuple

import pandas as pd

from haldxai.enrich.tables.loader import load_name2id, save_name2id
from haldxai.enrich.tables.utils import alloc_id

def _parse_e1_e2(tagged: Optional[str]) -> Tuple[Optional[str], Optional[str], str]:
    """
    解析 <e1>foo</e1> … <e2>bar</e2> 句子
    返回 (e1, e2, 去标签后的句子)
    """
    if not isinstance(tagged, str):
        return None, None, ""
    m1 = re.search(r"<e1>(.*?)</e1>", tagged, flags=re.I)
    m2 = re.search(r"<e2>(.*?)</e2>", tagged, flags=re.I)
    clean = re.sub(r"</?e[12]>", "", tagged, flags=re.I).strip()
    return (m1.group(1) if m1 else None,
            m2.group(1) if m2 else None,
            clean)

def build_relation_evidence(
    project_root: Path,
    df_llm_relationships: pd.DataFrame,
    df_pred_relations_llm: pd.DataFrame,
    df_pred_relations_articles: pd.DataFrame,
    *,
    force: bool = False
) -> pd.DataFrame:

    db_dir = project_root / "data/database"
    db_dir.mkdir(parents=True, exist_ok=True)

    output_csv = db_dir / "relation_evidence.csv"

    if output_csv.exists() and not force:
        print(f"🟡 relation_evidence.csv 已存在（跳过）。pass `force=True` 以重新生成。")
        return pd.read_csv(output_csv)

    print("▶ 构建 relation_evidence.csv …")

    # ── 0) 映射加载 ────────────────────────────────
    name2id = load_name2id(project_root)

    # ---------- 1) LLM 抽取 ----------
    llm_cols = ["pmid", "source_main_text", "target_main_text", "evidence"]
    llm = (
        df_llm_relationships[llm_cols]
        .rename(columns={
            "source_main_text": "source_entity_name",
            "target_main_text": "target_entity_name"
        })
        .assign(source="llm_extraction")
    )

    # ---------- 2) BERT-LM 预测 ----------
    bert_lm = df_pred_relations_llm[["input"]].copy()
    bert_lm[["source_entity_name",
             "target_entity_name",
             "evidence"]] = bert_lm["input"].apply(
        lambda t: pd.Series(_parse_e1_e2(t))
    )
    bert_lm = (
        bert_lm
        .assign(pmid="",
                source="bert_model_prediction")
        .drop(columns="input")
    )

    # ---------- 3) BERT-Articles 预测 ----------
    art = (
        df_pred_relations_articles[["pmid", "e1", "e2", "text"]]
        .rename(columns={
            "e1": "source_entity_name",
            "e2": "target_entity_name",
            "text": "evidence"
        })
        .assign(source="bert_model_prediction")
    )

    # ---------- 4) 合并 & 轻度清洗 ----------
    all_ev = pd.concat([llm, bert_lm, art], ignore_index=True)

    for col in ["source_entity_name", "target_entity_name"]:
        all_ev[col] = all_ev[col].astype(str).str.strip()

    all_ev["pmid"] = (
        pd.to_numeric(all_ev["pmid"], errors="coerce")
          .fillna(0).astype(int).astype(str).replace("0", "")
    )

    all_ev["source_entity_id"] = all_ev["source_entity_name"].apply(lambda n: alloc_id(name2id, n))
    all_ev["target_entity_id"] = all_ev["target_entity_name"].apply(lambda n: alloc_id(name2id, n))

    # 两端都拿得到 ID 才算有效
    all_ev = all_ev.dropna(subset=["source_entity_id", "target_entity_id"])

    # ---------- 6) relation_id & 主键 & 字段顺序 ----------
    # -------- 7. 生成 relation_id / PK --------
    all_ev["relation_id"] = (
        "Relation-" +
        all_ev["source_entity_id"].str.removeprefix("Entity-") +
        "-" +
        all_ev["target_entity_id"].str.removeprefix("Entity-")
    )

    all_ev.insert(0, "rel_ev_pk", range(1, len(all_ev) + 1))

    all_ev = all_ev[
        ["rel_ev_pk", "pmid", "relation_id",
         "source_entity_id", "target_entity_id",
         "source_entity_name", "target_entity_name",
         "evidence", "source"]
    ]

    # ------------- 5. 保存 ------------------
    all_ev.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✓ relation_evidence 写出 {len(all_ev):,} 行 → {output_csv}")

    save_name2id(project_root, name2id)  # ② 把可能新增的映射落盘
    print("✓ name2id.json 已更新，当前条数 =", len(name2id))

    return all_ev

