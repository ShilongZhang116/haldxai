# ─────────────────────────────────────────────────────────
#  relation.py   —— 生成 data/database/relations.csv
# ─────────────────────────────────────────────────────────
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict

import pandas as pd

from haldxai.enrich.tables.loader import load_name2id, load_id2name, save_name2id
from haldxai.enrich.tables.utils import alloc_id

_MODEL_LABEL_MAP: Dict[str, str] = {
    # DeepSeek
    "AgingRelated-DeepSeekR1-32B": "DeepSeekR1_32B",
    "AgingRelated-DeepSeekR1-7B":  "DeepSeekR1_7B",
    "AgingRelated-DeepSeekV3":     "DeepSeekV3",
    "JCRQ1-IF10-DeepSeekR1-32B":   "DeepSeekR1_32B",
    "JCRQ1-IF10-DeepSeekR1-7B":    "DeepSeekR1_7B",
    "JCRQ1-IF10-DeepSeekV3":       "DeepSeekV3",
    # SciSpacy
    "en_ner_bc5cdr_md":     "SciSpacy_BC5CDR",
    "en_ner_bionlp13cg_md": "SciSpacy_BioNLP13CG",
    "en_ner_jnlpba_md":     "SciSpacy_JNLPBA",
    None: "UnknownModel",
}


def _clean_src_file(fname: str) -> str:
    """hald_ageanno__xxx__std.csv →  Ageanno"""
    base = re.sub(r"^hald_|__std\.csv$", "", fname).split("__")[0]
    return re.sub(r"[^\w]", "_", base).capitalize()


def _model_to_label(model: str) -> str:
    """模型名 → 合法关系标签"""
    return re.sub(r"[^\w]", "_", str(model)).capitalize()


def build_relations(
        project_root: Path,
        df_ext_rels: pd.DataFrame,
        df_llm_relationships: pd.DataFrame,
        df_pred_relations_articles: pd.DataFrame,
        *,
        force: bool = False) -> pd.DataFrame:
    """
    Parameters
    ----------
    project_root : Path
    src          : dict   # loader.load_sources()
    out_dir      : Path | None (默认 <root>/data/database)
    force        : bool
    """
    db_dir = project_root / "data/database"
    db_dir.mkdir(parents=True, exist_ok=True)

    output_csv = db_dir / "relations.csv"

    if output_csv.exists() and not force:
        print(f"🟡 relations.csv 已存在（跳过）。pass `force=True` 以重新生成。")
        return pd.read_csv(output_csv)

    print("▶ 构建 relations.csv …")

    # ── 0) 映射加载 ────────────────────────────────
    name2id = load_name2id(project_root)

    # ───── ① 外部 DB 关系 ────────────────────────────
    ext_rels = (
        df_ext_rels
        .loc[:, ["source_name", "target_name", "source_file"]]
        .rename(columns={
            "source_name": "source_entity_name",
            "target_name": "target_entity_name"
        })
    )
    ext_rels[":TYPE"] = ext_rels["source_file"].map(_clean_src_file)

    # ───── ② LLM 标注关系 ───────────────────────────
    llm_rels = (
        df_llm_relationships
        .loc[:, ["source_main_text", "target_main_text", "model_name"]]
        .rename(columns={
            "source_main_text": "source_entity_name",
            "target_main_text": "target_entity_name"
        })
    )
    llm_rels[":TYPE"] = llm_rels["model_name"].map(_MODEL_LABEL_MAP).fillna("OtherModel")

    # ───── ③ 文章-BERT 预测关系 ─────────────────────
    art_rels = (
        df_pred_relations_articles
        .loc[:, ["e1", "e2"]]
        .rename(columns={"e1": "source_entity_name",
                         "e2": "target_entity_name"})
    )
    art_rels[":TYPE"] = "Bert_model_prediction"

    # ───── 合并三路 ─────────────────────────────────
    merged = pd.concat([ext_rels, llm_rels, art_rels], ignore_index=True)

    # 映射 ID
    merged[":START_ID"] = merged["source_entity_name"].apply(lambda n: alloc_id(name2id, n))
    merged[":END_ID"]   = merged["target_entity_name"].apply(lambda n: alloc_id(name2id, n))
    merged = merged.dropna(subset=[":START_ID", ":END_ID"])

    # 生成 relation_id
    merged["relation_id"] = (
        "Relation-" +
        merged[":START_ID"].str.replace("^Entity-", "", regex=True) + "-" +
        merged[":END_ID"].str.replace("^Entity-",   "", regex=True)
    )

    # 重排列
    cols = ["relation_id", ":START_ID", ":END_ID", "source_entity_name", "target_entity_name", ":TYPE"]

    # ---- ⑥ 保存 --------------------------------------------------------------
    merged.to_csv(output_csv, columns=cols, index=False, encoding="utf-8-sig")
    print(f"✓ relations 写出 {len(merged):,} 行 → {output_csv}")

    save_name2id(project_root, name2id)  # ② 把可能新增的映射落盘
    print("✓ name2id.json 已更新，当前条数 =", len(name2id))

    return merged

