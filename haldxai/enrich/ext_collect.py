"""
ext_collect.py
──────────────────────────────────────────────────────────────
把 data/external_db目录下的所有“已标准化” CSV
根据 node_source_config.json / relation_source_config.json
合成为：
  ▸ collected_nodes.csv      (entity_name, entity_type, …)
  ▸ collected_relations.csv  (source_name, target_name, …)
----------------------------------------------------------------
主入口:  build_collect(project_root: Path, *, force: bool=False)
"""

from __future__ import annotations
import json, csv, logging, sys, pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from haldxai.enrich.external_db.io_utils import read_tsv_robust                     # 你已有的工具
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,           # 明确指定到 stdout
    format="%(message)s"         # 只打印消息本身
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 0. 工具
# ──────────────────────────────────────────────────────────────
def _load_json(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────
# 1. 核心函数
# ──────────────────────────────────────────────────────────────
def build_collect(project_root: Path, *, force: bool = False) -> None:
    """把所有外部数据库节点/关系汇总到两个大 CSV"""

    # ------- 路径 -------
    ext_dir   = project_root / "data/external_db"
    out_dir   = project_root / "data/finals"
    cfg_dir   = project_root / "configs"
    node_cfg  = _load_json(cfg_dir / "node_source_config.json")
    rel_cfg   = _load_json(cfg_dir / "relation_source_config.json")

    out_nodes = out_dir / "collected_ext_nodes.csv"
    out_rels  = out_dir / "collected_ext_relations.csv"

    if not force and out_nodes.exists() and out_rels.exists():
        logger.warning("输出已存在，若需覆盖请传入 force=True")
        return

    # ================== 2. 汇总节点 ==================
    node_rows: list[dict] = []

    for cfg in node_cfg:
        csv_file = cfg["file"]
        logger.info(f"▶ 读取节点源: {csv_file}")

        key_col     = cfg["key_col"]
        primary_col = cfg.get("primary_col")
        extra_map   = cfg["extra_map"]

        usecols = [key_col] + ([primary_col] if primary_col else []) \
                + list(extra_map.values())
        if cfg["type_all_same"] == "False":
            usecols.append(cfg["entity_type_col"])

        df = read_tsv_robust(ext_dir / csv_file).dropna(subset=[key_col])

        # —— 列索引缓存 ——
        idx_key = df.columns.get_loc(key_col)
        idx_primary = df.columns.get_loc(primary_col) if primary_col else None
        idx_type = (df.columns.get_loc(cfg["entity_type_col"])
                    if cfg["type_all_same"] == "False" else None)
        extra_idx = {k: df.columns.get_loc(v) for k, v in extra_map.items()}

        for tup in df.itertuples(index=False, name=None):
            name = tup[idx_key]
            if not isinstance(name, str) or not name.strip():
                continue

            etype = (cfg["entity_type"]
                     if cfg["type_all_same"] == "True"
                     else str(tup[idx_type]).strip())

            primary = ("" if idx_primary is None
                       else ("" if pd.isna(tup[idx_primary])
                             else str(tup[idx_primary]).strip()))

            extra: Dict[str, Any] = {}
            for k, col in extra_idx.items():
                val = tup[col]
                if pd.isna(val) or val == "":
                    continue
                if isinstance(val, str) and (";" in val or "|" in val):
                    extra[k] = [v.strip() for v in val.replace("|", ";").split(";") if v.strip()]
                else:
                    extra[k] = val
            node_rows.append({
                "entity_name" : name.strip(),
                "entity_type" : etype,
                "primary_info": primary,
                "extra_json"  : _safe_json(extra) if extra else "{}",
                "source_file" : csv_file
            })

    df_nodes = pd.DataFrame(node_rows)
    logger.info(f"✓ 汇总节点 {len(df_nodes):,} 条")

    # ================== 3. 汇总关系 ==================
    rel_rows: list[dict] = []
    for cfg in rel_cfg:
        csv_file = cfg["file"]
        logger.info(f"▶ 读取关系源: {csv_file}")

        src_col, tgt_col = cfg["source_col"], cfg["target_col"]
        usecols = [src_col, tgt_col]
        if cfg["type_all_same"] == "False":
            usecols.append(cfg["relation_type_col"])

        df = read_tsv_robust(ext_dir / csv_file).dropna(subset=[src_col, tgt_col])

        idx_src = df.columns.get_loc(src_col)
        idx_tgt = df.columns.get_loc(tgt_col)
        idx_type = (df.columns.get_loc(cfg["relation_type_col"])
                    if cfg["type_all_same"] == "False" else None)

        for tup in df.itertuples(index=False, name=None):
            src, tgt = tup[idx_src], tup[idx_tgt]
            if not src or not tgt:
                continue
            rtype = (cfg["relation_type"] if cfg["type_all_same"] == "True"
                     else str(tup[idx_type]).strip())
            rel_rows.append({
                "source_name"  : str(src).strip(),
                "target_name"  : str(tgt).strip(),
                "relation_type": rtype,
                "source_file"  : csv_file
            })
    df_rels = pd.DataFrame(rel_rows)
    logger.info(f"✓ 汇总关系 {len(df_rels):,} 条")

    # ================== 4. 写文件 ==================
    df_nodes.to_csv(out_nodes, index=False, encoding="utf-8-sig",
                    quoting=csv.QUOTE_MINIMAL)
    df_rels.to_csv(out_rels,  index=False, encoding="utf-8-sig",
                    quoting=csv.QUOTE_MINIMAL)
    logger.info("🎉 完成写出：\n"
                f"    • {out_nodes}  ({len(df_nodes):,})\n"
                f"    • {out_rels}   ({len(df_rels):,})")


# ──────────────────────────────────────────────────────────────
# 2. Typer CLI 入口（可选）
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":              # 直接 python build_collect_std.py
    import typer, rich
    app = typer.Typer(pretty_exceptions_show_locals=False)

    @app.command("run")
    def _run(root: str = typer.Option(..., help="项目根目录"),
             force: bool = typer.Option(False, "--force", "-f",
                                        help="已存在时覆盖")):
        build_collect(Path(root), force=force)

    rich.print("[bold green]HALDxAI[/] collecting external nodes & relations …")
    app()
