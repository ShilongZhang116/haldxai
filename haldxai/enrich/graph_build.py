#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_unified_graph.py
──────────────────────────────────────────────────────────────
1. 读取 cache 中清洗后的 parquet：
   · collected_ext_nodes_clean.parquet
   · collected_ext_rels_clean.parquet
   · annotated_entities_clean.parquet
   · annotated_relationships_clean.parquet
2. 合并成
   · all_nodes.parquet      （去重并聚合 entity_type / source）
   · all_rels.parquet       （去重）
3. 依据 mappings/name2id.json 追加 entity_id / relation_id
4. 输出
   · all_nodes_with_id.parquet
   · all_rels_with_id.parquet
用法（Typer CLI）:
$ python -m haldxai.enrich.graph_build.build_unified_graph run --root F:/Project/HALDxAI-Suite/HALDxAI-Project
"""
from __future__ import annotations

import json, logging, pandas as pd
from pathlib import Path
import typer

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════
# 核心函数
# ════════════════════════════════════════════════════════════
def build_unified_graph(project_root: Path, *, force: bool = False) -> None:
    cache_dir   = project_root / "cache"
    map_dir     = project_root / "data/mappings"
    map_file    = map_dir / "name2id.json"

    # ---------- 输入 ----------
    f_ext_nodes = cache_dir / "collected_ext_nodes_clean.parquet"
    f_ext_rels  = cache_dir / "collected_ext_rels_clean.parquet"
    f_ann_ents  = cache_dir / "annotated_entities_clean.parquet"
    f_ann_rels  = cache_dir / "annotated_relationships_clean.parquet"

    f_all_nodes = cache_dir / "all_nodes_with_id.parquet"
    f_all_rels  = cache_dir / "all_rels_with_id.parquet"

    if not force and f_all_nodes.exists() and f_all_rels.exists():
        logger.warning("已存在 all_nodes_with_id / all_rels_with_id，跳过。"
                       "如需覆盖请加 --force")
        return

    logger.info("▶ 读取 parquet …")
    ext_nodes = pd.read_parquet(f_ext_nodes)
    ext_rels  = pd.read_parquet(f_ext_rels)
    ann_ents  = pd.read_parquet(f_ann_ents)
    ann_rels  = pd.read_parquet(f_ann_rels)

    # ---------- 合并节点 ----------
    logger.info("▶ 合并节点 …")
    ext_nodes_std = (
        ext_nodes.rename(columns={"entity_name": "entity_name",
                                  "entity_type": "entity_type",
                                  "source_file": "source"})
                  .loc[:, ["entity_name", "entity_type", "source"]]
    )
    ann_nodes_std = (
        ann_ents.rename(columns={"main_text": "entity_name",
                                 "entity_type": "entity_type"})
                 .loc[:, ["entity_name", "entity_type"]]
    )
    ann_nodes_std["source"] = "pubmed_article_llm"

    nodes_merged = pd.concat([ext_nodes_std, ann_nodes_std], ignore_index=True)

    all_nodes = (
        nodes_merged
        .groupby("entity_name", as_index=False)
        .agg({
            "entity_type": lambda x: ";".join(sorted(set(x.dropna()))),
            "source"     : lambda x: ";".join(sorted(set(x.dropna())))
        })
    )

    # ---------- 合并关系 ----------
    logger.info("▶ 合并关系 …")
    ext_rels_std = (
        ext_rels.rename(columns={"source_name": "source_entity_name",
                                 "target_name": "target_entity_name",
                                 "relation_type": "relation_type",
                                 "source_file": "source"})
                 .loc[:, ["source_entity_name", "target_entity_name",
                          "relation_type", "source"]]
    )
    ann_rels_std = (
        ann_rels.rename(columns={"source_main_text": "source_entity_name",
                                 "target_main_text": "target_entity_name",
                                 "relation_type": "relation_type"})
                 .loc[:, ["source_entity_name", "target_entity_name",
                          "relation_type"]]
    )
    ann_rels_std["source"] = "pubmed_article_llm"

    all_rels = (
        pd.concat([ext_rels_std, ann_rels_std], ignore_index=True)
          .drop_duplicates(subset=["source_entity_name",
                                   "target_entity_name",
                                   "relation_type"])
    )

    # ---------- 加载 name2id ----------
    with map_file.open(encoding="utf-8") as fh:
        name2id: dict[str, str] = json.load(fh)

    def map_name(n: str) -> str | pd.NA:
        return name2id.get(n, pd.NA)

    # ---------- 节点加 ID ----------
    all_nodes["entity_id"] = all_nodes["entity_name"].map(map_name)

    # ---------- 关系加 ID & 生成 relation_id ----------
    all_rels["source_entity_id"] = all_rels["source_entity_name"].map(map_name)
    all_rels["target_entity_id"] = all_rels["target_entity_name"].map(map_name)

    all_rels["relation_id"] = (
        "Relation-" +
        all_rels["source_entity_id"].str.replace("^Entity-", "", regex=True).astype(str) +
        "-" +
        all_rels["target_entity_id"].str.replace("^Entity-", "", regex=True).astype(str)
    )

    # ---------- 列顺序 ----------
    all_nodes = all_nodes[["entity_id", "entity_name", "entity_type", "source"]]
    all_rels  = all_rels[["relation_id", "source_entity_name", "target_entity_name",
                          "relation_type", "source",
                          "source_entity_id", "target_entity_id"]]

    # ---------- 保存 ----------
    all_nodes.to_parquet(f_all_nodes, index=False)
    all_rels.to_parquet(f_all_rels,   index=False)

    logger.info("🎉 完成写出：\n"
                f"    • {f_all_nodes}  ({len(all_nodes):,})\n"
                f"    • {f_all_rels}   ({len(all_rels):,})")


# ════════════════════════════════════════════════════════════
# Typer CLI
# ════════════════════════════════════════════════════════════
app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command("run")
def _run(root: str = typer.Option(..., help="项目根目录"),
         force: bool = typer.Option(False, "--force", "-f", help="覆盖已存在输出")):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    build_unified_graph(Path(root), force=force)

if __name__ == "__main__":
    app()
