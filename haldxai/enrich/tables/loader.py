# haldxai/tables/loader.py
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from typing import Dict

def load_sources(project_root: Path):
    """一次性读取 cache 里所有 parquet，返回命名元组"""
    cache = project_root / "cache"

    global Articles, Nodes, Rels, ExtNodes, ExtRels
    global LlmEnts, LlmRels, PredEnts, PredRelsLlm, PredRelsArt   # noqa: E501

    Articles       = pd.read_parquet(cache / "articles.parquet")
    Nodes          = pd.read_parquet(cache / "all_nodes_with_id.parquet")
    Rels           = pd.read_parquet(cache / "all_rels_with_id.parquet")
    ExtNodes       = pd.read_parquet(cache / "collected_ext_nodes_clean.parquet")
    ExtRels        = pd.read_parquet(cache / "collected_ext_rels_clean.parquet")
    LlmEnts        = pd.read_parquet(cache / "annotated_entities_clean.parquet")
    LlmRels        = pd.read_parquet(cache / "annotated_relationships_clean.parquet")
    PredEnts       = pd.read_parquet(cache / "predicted_entities.parquet")
    PredRelsLlm    = pd.read_parquet(cache / "predicted_relationships_from_llm.parquet")
    PredRelsArt    = pd.read_parquet(cache / "predicted_relationships_from_articles.parquet")

    print("✓ cache 数据全部载入完毕")

    # 返回 dict 方便解构
    return dict(
        Articles=Articles, Nodes=Nodes, Rels=Rels,
        ExtNodes=ExtNodes, ExtRels=ExtRels,
        LlmEnts=LlmEnts, LlmRels=LlmRels,
        PredEnts=PredEnts, PredRelsLlm=PredRelsLlm, PredRelsArt=PredRelsArt,
    )

def load_name2id(project_root: Path) -> Dict[str, str]:
    """读取 name2id.json 并统一小写"""
    n2i_path = project_root / "data/mappings/name2id.json"
    if not n2i_path.exists():
        print(f"⚠️  id2name.json 未找到：{n2i_path}")
        return {}
    with n2i_path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    return {k.lower(): v for k, v in raw.items()}


def load_id2name(project_root: Path) -> Dict[str, str]:
    i2n_path = project_root / "data/mappings/id2name.json"
    if not i2n_path.exists():
        print(f"⚠️  id2name.json 未找到：{i2n_path}")
        return {}
    with i2n_path.open(encoding="utf-8") as f:
        id2name = json.load(f)
    return {k: v for k, v in id2name.items() if k and v}


def save_name2id(project_root: Path, mapping: dict[str, str]) -> None:
    """把 name→id 映射保存成漂亮的 JSON"""
    n2i_path = project_root / "data/mappings/name2id.json"
    n2i_path.parent.mkdir(parents=True, exist_ok=True)
    n2i_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )