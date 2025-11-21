# haldxai/enrich/tables/relation_types.py
# ─────────────────────────────────────────────────────────
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from haldxai.enrich.tables.loader import load_name2id, save_name2id
from haldxai.enrich.tables.utils import alloc_id

# ────────────────────────────
# 内部工具
# ────────────────────────────
def _clean_source(src: str) -> str:
    """
    把 `hald_xxx__yyy__std.csv` → `xxx__yyy`
    """
    src = re.sub(r"^hald_", "", src)
    src = re.sub(r"__std\.csv$", "", src)
    return src

# ────────────────────────────
# 核心构建函数
# ────────────────────────────
def build_relation_types(
        project_root: Path,
        df_ext_rels: pd.DataFrame,
        df_llm_relationships: pd.DataFrame,
        df_pred_relations_llm: pd.DataFrame,
        df_pred_relations_articles: pd.DataFrame,
        *,
        force: bool = False
) -> pd.DataFrame:
    """
    Parameters
    ----------
    project_root : Path
        项目根目录
    src : dict
        `load_sources()` 返回的缓存对象
    out_dir : Path | None
        最终 CSV 输出目录；默认写入 `<project_root>/data/database`
    force : bool
        True 则覆盖已存在文件
    """
    db_dir = project_root / "data/database"
    db_dir.mkdir(parents=True, exist_ok=True)

    output_csv = db_dir / "relation_types.csv"

    if output_csv.exists() and not force:
        print(f"🟡 relation_types.csv 已存在（跳过）。pass `force=True` 以重新生成。")
        return pd.read_csv(output_csv)

    print("▶ 构建 relation_types.csv …")

    # ── 0) 映射加载 ────────────────────────────────
    name2id = load_name2id(project_root)

    # -------- 1. 外部标准库关系 --------
    ext = df_ext_rels[["source_name", "target_name",
                          "relation_type", "source_file"]].rename(
        columns={
            "source_name": "source_entity_name",
            "target_name": "target_entity_name",
            "source_file": "source"
        }
    )
    mask = ext["source"].str.startswith("hald_")
    ext.loc[mask, "source"] = ext.loc[mask, "source"].map(_clean_source)

    # -------- 2. 人工标注 LLM 关系 --------
    ann = df_llm_relationships[["source_main_text", "target_main_text",
                          "relation_type", "model_name"]].rename(
        columns={
            "source_main_text": "source_entity_name",
            "target_main_text": "target_entity_name",
            "model_name": "source"
        }
    )

    # -------- 3. BERT 预测关系（LLM prompts） --------
    bert_llm = df_pred_relations_llm[["input", "predicted_relation_type"]].copy()
    bert_llm["source"] = "bert_model_prediction"
    bert_llm[["source_entity_name", "target_entity_name"]] = bert_llm["input"].str.extract(
        r"<e1>(.*?)</e1>.*?<e2>(.*?)</e2>", expand=True
    )
    bert_llm = bert_llm.rename(columns={"predicted_relation_type": "relation_type"})[
        ["source_entity_name", "target_entity_name", "relation_type", "source"]
    ]

    # -------- 4. BERT 预测关系（全文） --------
    bert_art = df_pred_relations_articles[["e1", "e2", "predicted_relation_type"]].rename(
        columns={
            "e1": "source_entity_name",
            "e2": "target_entity_name",
            "predicted_relation_type": "relation_type"
        }
    )
    bert_art["source"] = "bert_model_prediction"

    # -------- 5. 合并去重 --------
    all_rels = pd.concat([ext, ann, bert_llm, bert_art], ignore_index=True)
    all_rels = all_rels.dropna(subset=["source_entity_name", "target_entity_name"])
    all_rels["source_entity_name"] = all_rels["source_entity_name"].str.strip()
    all_rels["target_entity_name"] = all_rels["target_entity_name"].str.strip()

    # -------- 6. 映射 entity_id --------
    all_rels["source_entity_id"] = all_rels["source_entity_name"].apply(lambda n: alloc_id(name2id, n))
    all_rels["target_entity_id"] = all_rels["target_entity_name"].apply(lambda n: alloc_id(name2id, n))

    all_rels = all_rels.dropna(subset=["source_entity_id", "target_entity_id"])

    # -------- 7. 生成 relation_id / PK --------
    all_rels["relation_id"] = (
        "Relation-" +
        all_rels["source_entity_id"].str.removeprefix("Entity-") +
        "-" +
        all_rels["target_entity_id"].str.removeprefix("Entity-")
    )

    all_rels.insert(0, "rel_pk", range(1, len(all_rels) + 1))

    cols = ["rel_pk", "relation_id",
            "source_entity_id", "target_entity_id",
            "source_entity_name", "target_entity_name",
            "relation_type", "source"]
    final_df = all_rels[cols]

    # ---- ⑥ 保存 --------------------------------------------------------------
    final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✓ entity_types_pred 写出 {len(final_df):,} 行 → {output_csv}")

    save_name2id(project_root, name2id)  # ② 把可能新增的映射落盘
    print("✓ name2id.json 已更新，当前条数 =", len(name2id))

    return final_df

