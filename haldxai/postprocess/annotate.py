"""
annotate.py  ‒ 将 DeepSeek / spaCy NER 结果结合 BioPortal 字典做实体 & 关系注释
───────────────────────────────────────────────────────────────────────────────
❶ 解析括号缩写            ❷ 清洗字符串            ❸ 读取 ner_output/*.csv
❹ 使用 final_entity_results.json 注入 canonical / synonyms / 定义等
❺ 保存到 data/ner_dictionary/annotated_*   (目录可自定义)

既可 CLI 也可 Notebook 调用：
from haldxai.postprocess.annotate import run_annotation
run_annotation(proj="F:/Project/HALDxAI",
                task="AgingRelated-DeepSeekV3",
                kind="deepseek",
                bio_dict="/my/final_entity_results.json",
                out_dir="D:/outputs")

kind: "deepseek" | "spacy"
"""

from __future__ import annotations
from pathlib import Path
import os, re, json, yaml, textwrap
import pandas as pd
from typing import List, Dict, Any

# ───────────────────────────────────────────
# 共用 regex & 清洗
# ───────────────────────────────────────────
_LEAD  = re.compile(r'^[^\w\u4e00-\u9fff]+')
_ONLY  = re.compile(r'[\d\W_]+$')

def _clean(s: str | float) -> str | None:
    if pd.isna(s):            # nan
        return None
    s = (str(s).replace(",", ";")
               .replace('"', '')
               .replace("'", '')
               .replace("\n", " ")
               .strip())
    s = _LEAD.sub("", s).lstrip()
    if not s or _ONLY.fullmatch(s):
        return None
    return s

# ───────────────────────────────────────────
# 括号缩写解析
# ───────────────────────────────────────────
def _looks_like_abbr(txt: str) -> bool:
    if not txt: return False
    if len(txt)==1 and txt.isalpha():
        return True
    if 2 <= len(txt)<=8:
        alpha = sum(c.isalpha() for c in txt)
        if alpha==0 or alpha/len(txt) < .5: return False
        if txt.isdigit(): return False
        return True
    return False

def parse_abbr(entity: str) -> dict:
    """返回 {original, main, details[]}"""
    details: List[Dict[str,str]] = []
    if "(" not in entity:
        return {"original": entity, "main": entity, "details": []}

    # 单尾括号模式
    m = re.match(r'^(.*?)\s*\(([^)]*)\)\s*$', entity)
    if m:
        pre, mid = m.group(1).strip(), m.group(2).strip()
        typ = "abbreviation" if _looks_like_abbr(mid) else "info"
        details.append({"content": mid, "type": typ})
        return {"original": entity, "main": pre or entity, "details": details}

    # 复杂多括号
    main_buf: List[str] = []
    for token in re.split(r'(\(.*?\))', entity):
        if token.startswith("(") and token.endswith(")"):
            c = token[1:-1].strip()
            typ = "abbreviation" if _looks_like_abbr(c) else "info"
            details.append({"content": c, "type": typ})
        else:
            if token.strip():
                main_buf.append(token)
    main = "".join(main_buf).strip() or entity
    return {"original": entity, "main": main, "details": details}

# ───────────────────────────────────────────
# 核心: 注释实体 & 关系
# ───────────────────────────────────────────
def _load_ner_csvs(base: Path, prefix: str) -> pd.DataFrame:
    files = sorted(p for p in base.glob(f"{prefix}_*.csv") if "Invalid" not in p.name)
    dfs   = [pd.read_csv(f) for f in files if f.stat().st_size]
    if not dfs:
        raise RuntimeError(f"未找到有效 {prefix}_*.csv in {base}")
    return pd.concat(dfs, ignore_index=True)

def _annotate_deepseek(proj: Path, task: str,
                       bio_dict: dict,
                       out_dir: Path):
    ner_dir = proj / "data/interim/ner_output" / task
    ents = _load_ner_csvs(ner_dir, "Entities")
    rels = _load_ner_csvs(ner_dir, "Relationships")

    # 1) 解析实体
    ent_rows = []
    for r in ents.itertuples(index=False):
        info = parse_abbr(r.entity_text)
        ent_rows.append({
            "batch_id": r.batch_id, "pmid": r.pmid,
            "entity_text": info["original"],
            "main_text": _clean(info["main"]),
            "details": json.dumps(info["details"], ensure_ascii=False),
            "entity_type": r.entity_type,
            "evidence": r.evidence,
            "evidence_valid": r.evidence_valid,
            "is_abbreviation": any(d["type"]=="abbreviation" for d in info["details"]),
        })
    ents_df = pd.DataFrame(ent_rows)

    # 2) 解析关系
    rel_rows = []
    for r in rels.itertuples(index=False):
        src = parse_abbr(r.source_entity)
        tgt = parse_abbr(r.target_entity)
        rel_rows.append({
            "batch_id": r.batch_id, "pmid": r.pmid,
            "source_entity_text": src["original"],
            "source_main_text": _clean(src["main"]),
            "source_details": json.dumps(src["details"], ensure_ascii=False),
            "target_entity_text": tgt["original"],
            "target_main_text": _clean(tgt["main"]),
            "target_details": json.dumps(tgt["details"], ensure_ascii=False),
            "relation_type": r.relation_type,
            "evidence": r.evidence,
            "evidence_valid": r.evidence_valid,
            "is_source_abbreviation": any(d["type"]=="abbreviation" for d in src["details"]),
            "is_target_abbreviation": any(d["type"]=="abbreviation" for d in tgt["details"]),
        })
    rels_df = pd.DataFrame(rel_rows)

    # 3) 注入 BioPortal 信息
    def _lookup(txt:str|None):
        if not txt: return {}
        return bio_dict.get(txt) or {}

    ents_df["entity_info"] = ents_df["main_text"].apply(_lookup)
    for col in ["source_main_text", "target_main_text",
                "source_entity_text", "target_entity_text"]:
        if col in rels_df.columns:
            rels_df[col.replace("_text","_info")] = rels_df[col].apply(_lookup)

    # 4) 保存
    ents_df.to_csv(out_dir/f"annotated_entities_{task}.csv",
                   index=False, encoding="utf-8-sig")
    rels_df.to_csv(out_dir/f"annotated_relationships_{task}.csv",
                   index=False, encoding="utf-8-sig")
    print(f"✅ {task} 注释完成 → {out_dir}")


def _annotate_spacy(proj: Path, task: str,
                    bio_dict: dict,
                    out_dir: Path):
    ner_dir = proj / "data/interim/ner_output/spacy" / task
    ents = _load_ner_csvs(ner_dir, "ner")   # 文件名 ner_2020.csv 等

    ent_rows = []
    for r in ents.itertuples(index=False):
        try:
            info = parse_abbr(r.entity_text)
            ent_rows.append({
                "pmid": r.pmid,
                "entity_text": info["original"],
                "main_text": _clean(info["main"]),
                "details": json.dumps(info["details"], ensure_ascii=False),
                "label": r.label,
                "is_abbreviation": any(d["type"]=="abbreviation" for d in info["details"]),
            })
        except TypeError:
            continue
    ents_df = pd.DataFrame(ent_rows)
    ents_df["entity_info"] = ents_df["main_text"].apply(lambda x: bio_dict.get(x) or {})
    ents_df.to_csv(out_dir/f"annotated_entities_{task}.csv",
                   index=False, encoding="utf-8-sig")
    print(f"✅ {task} 注释完成 → {out_dir}")


# ───────────────────────────────────────────
# 对外函数
# ───────────────────────────────────────────
def run_annotation(
    proj: str | Path,
    task: str,
    kind: str,                         # "deepseek" | "spacy"
    bio_dict: str | Path,
    out_dir: str | Path | None = None,
):
    """
    proj      : 项目根目录
    task      : 如 AgingRelated-DeepSeekV3 or en_ner_bc5cdr_md
    kind      : deepseek / spacy
    bio_dict  : final_entity_results.json
    out_dir   : 结果输出目录 (默认 project_root/data/ner_dict)
    """
    proj = Path(proj)
    out_dir = Path(out_dir or proj/"data/ner_dict")
    out_dir.mkdir(parents=True, exist_ok=True)

    bio_dict_obj = json.loads(Path(bio_dict).read_text(encoding="utf-8"))

    if kind == "deepseek":
        _annotate_deepseek(proj, task, bio_dict_obj, out_dir)
    elif kind == "spacy":
        _annotate_spacy(proj, task, bio_dict_obj, out_dir)
    else:
        raise ValueError("kind 只能是 deepseek / spacy")


# ───────────────────────────────────────────
# CLI
# ───────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys
    pa = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
        按任务批量注释实体/关系
        kind=deepseek ➜ 同时处理 Entities_* & Relationships_*
        kind=spacy    ➜ 只处理 spaCy 实体
        """))
    pa.add_argument("--proj", required=True)
    pa.add_argument("--task", required=True)
    pa.add_argument("--kind", choices=["deepseek","spacy"], required=True)
    pa.add_argument("--bio_dict", required=True)
    pa.add_argument("--out_dir")
    args = pa.parse_args()

    run_annotation(args.proj, args.task, args.kind,
                   args.bio_dict, args.out_dir)
