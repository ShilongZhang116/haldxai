from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

from haldxai.enrich.tables.loader import load_name2id, save_name2id
from haldxai.enrich.tables.utils import alloc_id


# ──────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────
def _clean_source(src: str) -> str:
    """把外部标准化 csv 名字缩短为 ageanno__... / hagr__..."""
    src = re.sub(r"^hald_", "", src)
    src = re.sub(r"__std\.csv$", "", src)
    return src


# ──────────────────────────────────────────────────────────────
# 核心入口
# ──────────────────────────────────────────────────────────────
def build_entity_types(
        project_root: Path,
        df_ext_nodes: pd.DataFrame,
        df_llm_entities: pd.DataFrame,
        df_pred_entities: pd.DataFrame,
        *,
        force: bool = False
) -> pd.DataFrame:

    db_dir = project_root / "data/database"
    db_dir.mkdir(parents=True, exist_ok=True)

    output_csv = db_dir / "entity_types.csv"

    if output_csv.exists() and not force:
        print(f"🟡 entity_types.csv 已存在（跳过）。pass `force=True` 以重新生成。")
        return pd.read_csv(output_csv)

    print("▶ 构建 entity_types.csv …")

    # ── 0) 映射加载 ────────────────────────────────
    name2id = load_name2id(project_root)

    # ---------- 1. 取各来源并统一列 ----------
    ext_nodes = (
        df_ext_nodes[["entity_name", "entity_type", "source_file"]]
        .rename(columns={"source_file": "source"})
    )

    llm_ents = (
        df_llm_entities[["main_text", "entity_type", "model_name"]]
        .rename(columns={"main_text": "entity_name", "model_name": "source"})
    )

    bert_pred = (
        df_pred_entities[["main_text", "predicted_type"]]
        .rename(columns={"main_text": "entity_name", "predicted_type": "entity_type"})
    )

    bert_pred["source"] = "bert_model_prediction"

    df_all = pd.concat([ext_nodes, llm_ents, bert_pred], ignore_index=True)
    df_all = df_all.dropna(subset=["source"])

    # ---------- 2. 规范化 source ----------
    mask_ext = df_all["source"].str.startswith("hald_")  # 只清洗外部 csv
    df_all.loc[mask_ext, "source"] = df_all.loc[mask_ext, "source"].map(_clean_source)

    # ---------- 3. 映射 entity_id ----------
    df_all["entity_id"] = df_all["entity_name"].apply(lambda n: alloc_id(name2id, n))

    # ---------- 4. 去重 ----------
    df_all = (
        df_all.drop_duplicates(subset=["entity_id", "entity_type", "source"])
        .reset_index(drop=True)
    )

    # ---------- 5. 自增主键 ----------
    df_all.insert(0, "etype_pk", range(1, len(df_all) + 1))

    # ------------- 5. 保存 ------------------
    df_all.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✓ entity_types 写出 {len(df_all):,} 行 → {output_csv}")

    save_name2id(project_root, name2id)  # ② 把可能新增的映射落盘
    print("✓ name2id.json 已更新，当前条数 =", len(name2id))

    return df_all