# ─────────────────────────────────────────────────────────
#  nodes.py   —— 生成 data/database/nodes.csv
# ─────────────────────────────────────────────────────────
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict

import pandas as pd

from haldxai.enrich.tables.loader import load_name2id, load_id2name, save_name2id
from haldxai.enrich.tables.utils import alloc_id

# ════════════════════════════════════════════════════════
# 映射、工具
# ════════════════════════════════════════════════════════
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
    """hald_ageanno__xxx__std.csv → Ageanno"""
    base = re.sub(r"^hald_|__std\.csv$", "", fname)
    return base.split("__")[0].capitalize()


def _merge_labels(series: pd.Series) -> str:
    """去重后用 | 连接；始终带 'Entity'"""
    return "|".join(sorted({"Entity", *series.dropna()}))


def build_nodes(
        project_root: Path,
        df_ext_nodes: pd.DataFrame,
        df_llm_entities: pd.DataFrame,
        *,
        force: bool = False
) -> pd.DataFrame:
    """
    Parameters
    ----------
    project_root : Path
    src : dict           # 来自 loader.load_sources()
    out_dir : Path | None (默认 <root>/data/database)
    force : bool
    """
    db_dir = project_root / "data/database"
    db_dir.mkdir(parents=True, exist_ok=True)

    output_csv = db_dir / "nodes.csv"

    if output_csv.exists() and not force:
        print(f"🟡 nodes.csv 已存在（跳过）。pass `force=True` 以重新生成。")
        return pd.read_csv(output_csv)

    print("▶ 构建 nodes.csv …")

    # ── 0) 映射加载 ────────────────────────────────
    name2id = load_name2id(project_root)

    id2name = load_id2name(project_root)


    # ── 外部 DB 节点 ───────────────────────────────
    ext_nodes = (
        df_ext_nodes
        .loc[:, ["entity_name", "source_file"]]
        .assign(
            src_label=lambda d: d["source_file"].map(_clean_src_file),
            entity_id=lambda d: d["entity_name"].apply(lambda x: alloc_id(name2id, x))
        )
        .dropna(subset=["entity_id"])[["entity_id", "entity_name", "src_label"]]
    )

    # ── LLM 标注节点 ───────────────────────────────
    llm_nodes = (
        df_llm_entities
        .loc[:, ["main_text", "model_name"]]
        .rename(columns={"main_text": "entity_name"})
        .assign(
            src_label=lambda d: d["model_name"].map(_MODEL_LABEL_MAP).fillna("OtherModel"),
            entity_id=lambda d: d["entity_name"].apply(lambda x: alloc_id(name2id, x))
        )
        .dropna(subset=["entity_id"])[["entity_id", "entity_name", "src_label"]]
    )

    merged = pd.concat([ext_nodes, llm_nodes], ignore_index=True)

    nodes_tmp = (
        merged
        .groupby("entity_id", as_index=False)
        .agg(name=("entity_name", "first"),
             label=("src_label", _merge_labels))
    )

    # 2️⃣  用 map 替换：有首选名就用，没有就保持原值
    nodes_final = nodes_tmp.assign(
        name=lambda d: d["entity_id"].map(id2name).fillna(d["name"])
    )

    nodes_final.rename(columns={
        "entity_id": "node_id:ID",
        "label": ":LABEL"
    }, inplace=True)

    # ---- ⑥ 保存 --------------------------------------------------------------
    nodes_final.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✓ nodes 写出 {len(nodes_final):,} 行 → {output_csv}")

    save_name2id(project_root, name2id)  # ② 把可能新增的映射落盘
    print("✓ name2id.json 已更新，当前条数 =", len(name2id))

    return nodes_final