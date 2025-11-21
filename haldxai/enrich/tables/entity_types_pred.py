# haldxai/enrich/tables/entity_types_pred.py
# ─────────────────────────────────────────────────────────────
"""build_entity_types_pred

把 `src["PredEnts"]` 中的实体类型预测结果整理为可入库 CSV。

输出字段
--------
pred_pk | pmid | entity_id | entity_name
| predicted_type | similarity
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from haldxai.enrich.tables.loader import load_name2id, save_name2id
from haldxai.enrich.tables.utils import alloc_id


def build_entity_types_pred(
    project_root: Path,
    df_pred_entities: pd.DataFrame,
    *,
    force: bool = False
) -> pd.DataFrame:
    """
    Parameters
    ----------
    project_root : Path
        项目根目录，用于定位 cache/mapping、输出目录等。
    df_pred_ents : pd.DataFrame
        loader 返回的 `src["PredEnts"]`，至少要有：
        pmid | main_text | predicted_type | similarity
    name2id : dict[str,str] | None
        已经加载好的映射表（可选；不传则自动从 cache 读取）
    """

    db_dir = project_root / "data/database"
    db_dir.mkdir(parents=True, exist_ok=True)

    output_csv = db_dir / "entity_types_pred.csv"

    if output_csv.exists() and not force:
        print(f"🟡 entity_types_pred.csv 已存在（跳过）。pass `force=True` 以重新生成。")
        return pd.read_csv(output_csv)

    print("▶ 构建 entity_types_pred.csv …")

    # ── 0) 映射加载 ────────────────────────────────
    name2id = load_name2id(project_root)

    # ---- ① 选列并重命名 ----------------------------------------------------
    cols_needed = ["pmid", "main_text", "predicted_type", "similarity"]
    df = df_pred_entities.loc[:, cols_needed].rename(
        columns={"main_text": "entity_name"}
    ).copy()

    # ---- ② PMID 规范化 ------------------------------------------------------
    df["pmid"] = (
        pd.to_numeric(df["pmid"], errors="coerce")
        .fillna(0).astype(int).astype(str).replace("0", "")
    )

    # ---- ③ entity_id 映射 ---------------------------------------------------
    df["entity_id"] = df["entity_name"].apply(lambda n: alloc_id(name2id, n))

    # ---- ④ 添加自增主键 ------------------------------------------------------
    df.insert(0, "pred_pk", range(1, len(df) + 1))

    # ---- ⑤ 字段顺序固定 ------------------------------------------------------
    df = df[
        [
            "pred_pk",
            "pmid",
            "entity_id",
            "entity_name",
            "predicted_type",
            "similarity",
        ]
    ]

    # ---- ⑥ 保存 --------------------------------------------------------------
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✓ entity_types_pred 写出 {len(df):,} 行 → {output_csv}")

    save_name2id(project_root, name2id)  # ② 把可能新增的映射落盘
    print("✓ name2id.json 已更新，当前条数 =", len(name2id))

    return df

