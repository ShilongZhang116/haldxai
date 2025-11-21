"""
postprocess.py (v2)
-------------------
✓ DeepSeek-LLM / spaCy NER → 实体词典
✓ 输入 / 输出目录可以显式传参
"""

from __future__ import annotations
from pathlib import Path
import os, json, yaml
import pandas as pd

from .abbrev import parse_entity_abbreviation

# ---------- 默认路径（可被参数覆盖） ----------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CFG           = yaml.safe_load((_PROJECT_ROOT / "configs/config.yaml").read_text(encoding="utf-8"))
_DEFAULT_DICT_DIR = _PROJECT_ROOT / "data/ner_dict"
_DEFAULT_DICT_DIR.mkdir(exist_ok=True, parents=True)


# ---------- 通用工具 ---------- #
def _concat_csv(files: list[Path]) -> pd.DataFrame:
    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"⚠️ 读取失败，跳过 {fp.name}：{e}")
    if not dfs:
        raise RuntimeError("❌ 没有可用 CSV！")
    return pd.concat(dfs, ignore_index=True)


def _parse_unique_entities(texts) -> pd.DataFrame:
    seen, rows = set(), []
    for t in texts:
        if pd.isna(t) or t in seen:
            continue
        seen.add(t)
        parsed = parse_entity_abbreviation(str(t))
        rows.append({
            "original_text":  parsed["original_text"],
            "main_text":      parsed["main_text"],
            "details":        json.dumps(parsed["details"], ensure_ascii=False),
            "is_abbreviation": any(d["type"] == "abbreviation" for d in parsed["details"]),
        })
    return pd.DataFrame(rows)


# ---------- DeepSeek-LLM ---------- #
def build_deepseek_entity_dict(
    task_name: str,
    ner_output_dir: Path | str | None = None,
    dict_out_dir:  Path | str | None = None,
) -> Path:
    """
    参数
    ----
    task_name      : 例 "AgingRelated-DeepSeekV3"
    ner_output_dir : 若为空 → <project_root>/data/ner_output/<task_name>
    dict_out_dir   : 若为空 → <project_root>/data/ner_dictionary
    """
    ner_dir  = Path(ner_output_dir or _PROJECT_ROOT / f"data/interim/ner_output/{task_name}")
    out_dir  = Path(dict_out_dir  or _DEFAULT_DICT_DIR)
    out_dir.mkdir(exist_ok=True, parents=True)
    out_fp   = out_dir / f"parsed_entities_{task_name}.csv"

    entity_files   = sorted(ner_dir.glob("Entities_*.csv"))
    relation_files = sorted(ner_dir.glob("Relationships_*.csv"))

    ents_df  = _concat_csv(entity_files)
    rels_df  = _concat_csv(relation_files)

    all_txt = pd.concat([
        ents_df["entity_text"],
        rels_df["source_entity"],
        rels_df["target_entity"],
    ], ignore_index=True).dropna().unique()

    _parse_unique_entities(all_txt).to_csv(out_fp, index=False, encoding="utf-8-sig")
    print(f"✅ DeepSeek 词典 → {out_fp}")
    return out_fp


# ---------- spaCy ---------- #
def build_spacy_entity_dict(
    model_name: str,
    ner_output_dir: Path | str | None = None,
    dict_out_dir:  Path | str | None = None,
) -> Path:
    """
    model_name 例 "en_ner_bc5cdr_md"
    """
    ner_dir = Path(ner_output_dir or _PROJECT_ROOT / f"data/interim/ner_output/spacy/{model_name}")
    out_dir = Path(dict_out_dir  or _DEFAULT_DICT_DIR)
    out_dir.mkdir(exist_ok=True, parents=True)
    out_fp  = out_dir / f"parsed_entities_{model_name}.csv"

    ents_df = _concat_csv(sorted(ner_dir.glob("ner_*.csv")))

    _parse_unique_entities(ents_df["entity_text"].dropna().unique())\
        .to_csv(out_fp, index=False, encoding="utf-8-sig")
    print(f"✅ spaCy 词典   → {out_fp}")
    return out_fp
