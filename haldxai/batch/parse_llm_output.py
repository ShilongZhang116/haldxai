"""
解析 LLM NER 结果 → 抽实体 & 关系，并保存年度 CSV

✓ 断点友好：按批次、按年份增量保存
✓ 与 run_llm_batch 写入的 result_*.jsonl 完全兼容
✓ 读取全局 config.yaml / prompts.yaml，变量全部沿用你的新风格
"""

from __future__ import annotations
from pathlib import Path
import re, json, jsonlines, yaml, rich
import pandas as pd
from typing import Tuple, List, Dict

# ────────────────────── 公共工具 ────────────────────── #
def _extract_year(fname: str) -> int | None:
    m = re.search(r"Y(\d{4})", fname)
    return int(m.group(1)) if m else None

def _normalize(text: str) -> str:
    """简易归一化：小写 + 去标点 + 多空格合并"""
    import re
    t = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", t).strip()

def _evidence_in_abs(ev: str, abs_: str) -> bool:
    return _normalize(ev) in _normalize(abs_)

def _extract_json(content: str):
    """
    支持三种返回格式：
    1. 直接 dict / list（OpenAI function call）
    2. ```json\n{...}\n``` 区块
    3. 普通文本（直接返回）
    """
    if isinstance(content, dict):
        return content
    import re, json
    m = re.search(r"```json\s*(.*?)\s*```", content, flags=re.S | re.I)
    try:
        return json.loads(m.group(1) if m else content)
    except json.JSONDecodeError:
        return {}

def _locate_pmid_csv(bid: str, dir_: Path) -> Path:
    for suffix in (".meta.csv", "_ner_input.csv", ".csv"):
        fp = dir_ / f"{bid}{suffix}"
        if fp.exists():
            return fp
    raise FileNotFoundError(f"未找到 PMID 索引文件: {bid} (.meta.csv / _ner_input.csv)")

# ────────────────────── 主解析函数 ────────────────────── #
def parse_one_batch(
    batch_id: str,
    article_info_dir : Path,
    batch_input_dir  : Path,
    batch_result_dir : Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回 (实体 DF, 关系 DF)
    ─────────────────────────────────────────
    前置约定：
        - 文章信息 CSV：        <article_info_dir>/<batch_id>.csv
        - 输入 PMIDs 索引表：   <batch_input_dir>/<batch_id>.meta.csv
        - LLM 结果：           <batch_result_dir>/result_<batch_id>.jsonl
    """
    art_csv   = article_info_dir / f"{batch_id}.csv"
    pmid_csv = _locate_pmid_csv(batch_id, batch_input_dir)
    jsonl_fp  = batch_result_dir / f"result_{batch_id}.jsonl"

    if not jsonl_fp.exists():
        rich.print(f"[yellow]⚠️ {jsonl_fp} 不存在，跳过[/yellow]")
        return pd.DataFrame(), pd.DataFrame()

    # --- 加载 ---
    df_art   = pd.read_csv(art_csv,  dtype=str)
    df_pmids = pd.read_csv(pmid_csv, dtype=str)
    pmid2abs = dict(zip(df_art.pmid.astype(str), df_art.abstract.astype(str)))

    ents, rels = [], []

    with jsonlines.open(jsonl_fp) as r:
        for idx, obj in enumerate(r):
            req   = obj["request"]
            rsp   = obj["response"]["choices"][0]["message"]["content"]
            pmid  = df_pmids.loc[idx, "pmid"]
            abs_  = pmid2abs.get(str(pmid), "")

            js = _extract_json(rsp)

            # ---------- 实体 ----------
            for ent in js.get("entities", []):
                ev = ent.get("evidence", "")
                ents.append(
                    dict(
                        batch_id=batch_id,
                        pmid=pmid,
                        entity_text = ent.get("entity_text", ""),
                        entity_type = ent.get("entity_type", ""),
                        evidence    = ev,
                        evidence_valid = _evidence_in_abs(ev, abs_),
                    )
                )

            # ---------- 关系 ----------
            for rel in js.get("relationships", []):
                ev = rel.get("evidence", "")
                rels.append(
                    dict(
                        batch_id=batch_id,
                        pmid=pmid,
                        source_entity = rel.get("source_entity",  rel.get("source_entity_id","")),
                        target_entity = rel.get("target_entity",  rel.get("target_entity_id","")),
                        relation_type = rel.get("relation_type", ""),
                        confidence    = rel.get("confidence", ""),
                        evidence      = ev,
                        evidence_valid = _evidence_in_abs(ev, abs_),
                    )
                )

    return pd.DataFrame(ents), pd.DataFrame(rels)

# ────────────────────── 按年份聚合并保存 ────────────────────── #
def parse_by_year(
    task_name: str,
    project_root: Path,
    article_info_dir : Path,
    batch_input_dir  : Path,
    batch_result_dir : Path,
    save_dir         : Path,
    years: List[int] | None = None,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    all_jsonl = sorted(p for p in batch_result_dir.glob("result_*.jsonl"))
    if years:
        all_jsonl = [p for p in all_jsonl if _extract_year(p.name) in years]
    if not all_jsonl:
        rich.print("[yellow]⚠️ 未找到符合条件的 result_*.jsonl[/yellow]")
        return

    # —— 收集同一年份的 batch_id —— #
    year2batch: Dict[int, List[str]] = {}
    for p in all_jsonl:
        yr = _extract_year(p.name)
        bid = p.stem.replace("result_", "")
        year2batch.setdefault(yr, []).append(bid)

    for yr, batch_ids in year2batch.items():
        rich.print(f"[cyan]⏩ 解析 {yr}: {len(batch_ids)} 个批次[/cyan]")

        ents_all, rels_all = [], []
        for bid in batch_ids:
            ent_df, rel_df = parse_one_batch(
                bid, article_info_dir, batch_input_dir, batch_result_dir
            )
            ents_all.append(ent_df)
            rels_all.append(rel_df)

        df_ents = pd.concat(ents_all, ignore_index=True) if ents_all else pd.DataFrame()
        df_rels = pd.concat(rels_all, ignore_index=True) if rels_all else pd.DataFrame()

        # —— 保存 —— #
        ent_csv = save_dir / f"Entities_Y{yr}_{task_name}.csv"
        rel_csv = save_dir / f"Relationships_Y{yr}_{task_name}.csv"
        df_ents.to_csv(ent_csv, index=False, encoding="utf-8-sig")
        df_rels.to_csv(rel_csv, index=False, encoding="utf-8-sig")

        rich.print(f"  ✅ 实体 {len(df_ents):,} 行 → {ent_csv.name}")
        rich.print(f"  ✅ 关系 {len(df_rels):,} 行 → {rel_csv.name}")

# ────────────────────── CLI 入口 ────────────────────── #
if __name__ == "__main__":
    import argparse, sys
    pa = argparse.ArgumentParser()
    pa.add_argument("--task", required=True, help="如 JCRQ1-IF10-DeepSeekV3")
    pa.add_argument("--years", nargs="*", type=int, default=[])
    args = pa.parse_args()

    root = Path(__file__).resolve().parents[2]      # HALDxAI-Project 根目录
    cfg  = yaml.safe_load((root/"configs/config.yaml").read_text(encoding="utf-8"))

    task  = args.task
    art_dir  = root / f"data/batch_process/batch_articles_info/{task}"
    in_dir   = root / f"data/batch_process/batch_llm_ner_input/{task}"
    res_dir  = root / f"data/batch_process/batch_results/{task}"
    save_dir = root / f"data/ner_output/{task}"

    try:
        parse_by_year(
            task_name  = task,
            project_root= root,
            article_info_dir = art_dir,
            batch_input_dir  = in_dir,
            batch_result_dir = res_dir,
            save_dir         = save_dir,
            years            = args.years or None,
        )
    except Exception as e:
        rich.print(f"[red]❌ 解析失败: {e}[/red]")
        sys.exit(1)
