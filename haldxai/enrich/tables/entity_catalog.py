# haldxai/tables/entity_catalog.py
from __future__ import annotations
from pathlib import Path
import json, hashlib, pandas as pd

from haldxai.enrich.tables.loader import load_name2id, save_name2id
from haldxai.enrich.tables.utils import alloc_id

def _pg_array(elems: list[str]) -> str:
    if not elems:
        return "{}"

    def esc(s: str) -> str:
        # 1) 先处理反斜杠
        s = s.replace("\\", "\\\\")
        # 2) 再处理双引号
        s = s.replace('"', '\\"')
        # 3) 外层加引号
        return f'"{s}"'
    return "{" + ",".join(esc(e) for e in elems) + "}"

def build_entity_catalog(
        project_root: Path,
        *,
        force: bool = False
) -> pd.DataFrame:
    """
    根据 LLM 注释实体 + BioPortal 结果，汇总成 entity_catalog
    """

    db_dir      = project_root / "data/database"
    db_dir.mkdir(parents=True, exist_ok=True)

    output_csv = db_dir / "entity_catalog.csv"

    if output_csv.exists() and not force:
        print(f"🟡 entity_catalog.csv 已存在（跳过）。pass `force=True` 以重新生成。")
        return pd.read_csv(output_csv)

    print("▶ 构建 entity_catalog.csv …")

    # ------------------------------------------------------------------ #
    # 1. 读取 name2id 映射
    # ------------------------------------------------------------------ #
    name2id = load_name2id(project_root)

    # ------------------------------------------------------------------ #
    # 2. 解析 BioPortal info
    # ------------------------------------------------------------------ #
    bp_json = project_root / "data/ner_dict/bioPortal/final_entity_results.json"
    bp_map  = json.loads(bp_json.read_text(encoding="utf-8"))

    rows: list[dict] = []
    for name, info in bp_map.items():
        items = info.get("search_result", {}).get("items", [])
        for it in items:
            rows.append({
                "entity_name":   name,
                "pref_label":    it.get("pref_label"),
                "definitions":   (it.get("definitions") or [None])[0],
                "synonyms":      it.get("synonyms") or [],
                "ontology":      it.get("ontology"),
                "class_iri":     it.get("class_iri"),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("⚠ 无有效 BioPortal 记录，返回空表")
        return df

    # ------------------------------------------------------------------ #
    # 3. 计算 entity_id
    # ------------------------------------------------------------------ #
    df["entity_id"] = df["entity_name"].apply(lambda n: alloc_id(name2id, n))

    # ------------------------------------------------------------------ #
    # 4. 字段聚合
    # ------------------------------------------------------------------ #
    clean = lambda seq: [s for s in seq if isinstance(s, str) and s.strip()]

    out = (
        df.groupby("entity_id")
        .agg(
            entity_name=("entity_name", "first"),
            pref_label=("pref_label", lambda x: ";".join(sorted(clean(x)))),
            definitions=("definitions", lambda x: ";".join(sorted(clean(x)))),
            synonyms=("synonyms",
                      lambda col: _pg_array(
                          sorted(clean(
                              y for lst in col if isinstance(lst, list)
                              for y in lst
                          ))
                      )),
            ontology=("ontology", lambda x: ";".join(sorted(clean(x)))),
            class_iri=("class_iri", lambda x: ";".join(sorted(clean(x))))
        )
        .reset_index()
    )

    # 增加主键
    out.insert(0, "pk", range(1, len(out) + 1))

    save_name2id(project_root, name2id)  # ② 把可能新增的映射落盘
    print("✓ name2id.json 已更新，当前条数 =", len(name2id))

    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✓ entity_catalog 写出 {len(out):,} 行 → {output_csv}")

    return out