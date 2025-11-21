# haldxai/postprocess/build_id_mapping.py
# ============================================================
"""HALD 统一实体 ID 映射生成器
--------------------------------------------------------------
输入
  • data/finals/collected_ext_nodes.csv
  • data/finals/collected_ext_relations.csv
  • data/finals/all_annotated_entities.csv
  • data/finals/all_annotated_relationships.csv
  • config/node_source_config.json            # 用于解析同义词列
输出
  • cache/name2id.csv / .json
  • cache/id2name.csv  / .json
用法
  # 直接函数调用
from haldxai.postprocess.build_id_mapping import build_id_mapping
build_id_mapping("F:/Project/HALDxAI-Suite/HALDxAI-Project", force=True)

  # CLI
  python -m haldxai.postprocess.build_id_mapping run \
         --root F:/Project/HALDxAI-Suite/HALDxAI-Project --force
"""
from __future__ import annotations
import json, re, hashlib, csv, logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Any, Dict, List

import pandas as pd
import typer, rich

# ---------- 日志 ----------
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s: %(message)s")
log = logging.getLogger("id-map")

# ---------- 通用 & 正则 ----------
PAT_SYN = re.compile(r'(gene_alias(_description)?$|synonyms$|external_synonyms$|entity_synonyms$)', re.I)
canonical = lambda s: str(s).strip().upper()
make_node_id = lambda text: "Entity-" + hashlib.md5(text.encode("utf-8")).hexdigest()[:10]

# ---------- 读取工具 ----------
def _load_csv(fp: Path, **kw) -> pd.DataFrame:
    """尝试两种引号方案，尽量鲁棒"""
    for qchar in ('"', "'"):
        try:
            return pd.read_csv(fp, dtype=str, low_memory=False,
                               quoting=csv.QUOTE_MINIMAL, quotechar=qchar, **kw)
        except Exception:
            continue
    # 再次 fallback
    return pd.read_csv(fp, dtype=str, low_memory=False, **kw)

def _load_parquet(fp: Path) -> pd.DataFrame:
    return pd.read_parquet(fp) if fp.exists() else pd.DataFrame()

# ========================== 核心 ========================== #
def build_id_mapping(project_root: str | Path, *, force: bool = False) -> None:
    root      = Path(project_root)
    finals_dir    = root / "data/finals"
    ext_dir   = root / "data/external_db"
    cfg_dir   = root / "configs"
    cache_dir = root / "cache"
    mappings_dir = root / "data/mappings"
    mappings_dir.mkdir(exist_ok=True, parents=True)

    out_name2id = mappings_dir / "name2id.csv"
    out_id2name = mappings_dir / "id2name.csv"

    if not force and out_name2id.exists() and out_id2name.exists():
        log.warning("映射已存在，跳过。若需重建请加 --force")
        return

    # ---------- 1. 同义词字典（来自 node_source_config） ----------
    syn_rows: List[Dict[str, str]] = []
    node_cfg: List[Dict[str, Any]] = json.loads((cfg_dir / "node_source_config.json").read_text(encoding="utf-8"))

    for cfg in node_cfg:
        fp      = ext_dir / cfg["file"]
        key_col = cfg["key_col"]
        df      = _load_csv(fp)

        # 同义词列
        syn_cols = [c for c in df.columns if PAT_SYN.search(c)]
        if not syn_cols:
            continue

        for _, row in df.iterrows():
            primary = str(row.get(key_col, "")).strip()
            if not primary:                 # 无主名跳过
                continue

            syn_set = {primary}
            for sc in syn_cols:
                raw = str(row.get(sc, "")).strip()
                if raw and raw.lower() not in {"nan", "none"}:
                    parts = re.split(r'[;|]', raw)
                    syn_set.update(p.strip() for p in parts if p.strip())

            for syn in syn_set:
                syn_rows.append({
                    "synonym"      : syn,
                    "primary_name" : primary,
                    "source_table" : cfg["file"]
                })

    df_syn = (pd.DataFrame(syn_rows)
                .drop_duplicates()
                .assign(canonical_primary=lambda d: d["primary_name"].map(canonical)))

    # ---------- 2. 历史 “所有出现过的实体写法” ----------
    def _safe_read(name):  # helper
        fp = mappings_dir / name
        return _load_csv(fp) if fp.exists() else pd.DataFrame()

    df_ext_nodes = _load_parquet(cache_dir / "collected_ext_nodes_clean_entity_string.parquet")
    df_ext_rels = _load_parquet(cache_dir / "collected_ext_relations_clean_entity_string.parquet")
    df_llm_ents = _load_parquet(cache_dir / "all_annotated_entities_clean_entity_string.parquet")
    df_llm_rels = _load_parquet(cache_dir / "all_annotated_relationships_clean_entity_string.parquet")

    # 2) 如果 cache 里为空，再 fallback 到 finals/csv（可选）
    def _fallback_csv(name: str) -> pd.DataFrame:
        fp = finals_dir / name
        return _load_csv(fp) if fp.exists() else pd.DataFrame()

    if df_ext_nodes.empty: df_ext_nodes = _fallback_csv("collected_ext_nodes.csv")
    if df_ext_rels.empty:  df_ext_rels = _fallback_csv("collected_ext_relations.csv")
    if df_llm_ents.empty:  df_llm_ents = _fallback_csv("all_annotated_entities.csv")
    if df_llm_rels.empty:  df_llm_rels = _fallback_csv("all_annotated_relationships.csv")

    all_terms = pd.concat([
        df_ext_nodes.get("entity_name", pd.Series(dtype=str)),
        df_ext_rels.get("source_name", pd.Series(dtype=str)),
        df_ext_rels.get("target_name", pd.Series(dtype=str)),
        df_llm_ents.get("main_text", pd.Series(dtype=str)),
        df_llm_rels.get("source_main_text", pd.Series(dtype=str)),
        df_llm_rels.get("target_main_text", pd.Series(dtype=str)),
    ], ignore_index=True).dropna()

    # ---------- 3. 分配 ID ----------
    canonical_to_id: Dict[str, str] = {
        cp: make_node_id(cp) for cp in df_syn["canonical_primary"].unique()
    }
    HALD_NAME2ID: Dict[str, str] = {}
    HALD_ID2NAME: Dict[str, str] = {}

    # 3-a 同义词先入
    for syn, prim, can in df_syn[["synonym", "primary_name", "canonical_primary"]].itertuples(index=False):
        eid = canonical_to_id[can]
        HALD_NAME2ID[syn] = eid
        HALD_ID2NAME.setdefault(eid, prim)   # 主名优先

    # 3-b 历史写法
    name_counter: Dict[str, Counter] = defaultdict(Counter)
    for t in all_terms:
        t = str(t).strip()
        if not t:
            continue
        c = canonical(t)
        name_counter[c][t] += 1

    for can, cnt in name_counter.items():
        eid = canonical_to_id.setdefault(can, make_node_id(can))
        # 写法映射
        for variant in cnt:
            HALD_NAME2ID.setdefault(variant, eid)
        # 若还没有优选写法 → 选出现频次最高的
        HALD_ID2NAME.setdefault(eid, cnt.most_common(1)[0][0])

    log.info(f"✓ NAME→ID 映射数: {len(HALD_NAME2ID):,}")
    log.info(f"✓ ID→NAME 映射数: {len(HALD_ID2NAME):,}")

    # ---------- 4. 输出 ----------
    pd.DataFrame(HALD_NAME2ID.items(), columns=["name", "entity_id"])\
      .to_csv(out_name2id, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(HALD_ID2NAME.items(),  columns=["entity_id", "best_name"])\
      .to_csv(out_id2name, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    # 额外 JSON（可选）
    (mappings_dir / "name2id.json").write_text(json.dumps(HALD_NAME2ID, ensure_ascii=False, indent=2), encoding="utf-8")
    (mappings_dir / "id2name.json").write_text(json.dumps(HALD_ID2NAME, ensure_ascii=False, indent=2),  encoding="utf-8")

    rich.print(f"[bold green]🎉 ID 映射已生成[/]\n"
               f"  • {out_name2id}\n  • {out_id2name}")

# ========================== Typer CLI ========================== #
app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command("run")
def _run(root: str = typer.Option(..., help="项目根目录"),
         force: bool = typer.Option(False, "--force/-f", help="覆盖已有输出")):
    """生成 HALD 统一实体 ID 映射"""
    build_id_mapping(root, force=force)

if __name__ == "__main__":
    app()
