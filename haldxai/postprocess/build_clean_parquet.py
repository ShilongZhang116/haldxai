# build_clean_parquet.py
# --------------------------------------------------------
"""
清洗 collected_ext_* / all_annotated_* 字符串列
并写成 parquet 供下游检索 / 特征工程使用
"""

from __future__ import annotations
import pandas as pd, logging
from pathlib import Path
import typer, rich, sys

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,           # 明确指定到 stdout
    format="%(message)s"         # 只打印消息本身
)

# ---------- 基础清洗 ----------
def _clean(s):
    if pd.isna(s):
        return s
    return (str(s)
            .replace(",", ";")
            .replace('"', "")
            .replace("'", "")
            .replace("\n", " ")
            .strip())

def _clean_df(df: pd.DataFrame, cols: list[str]):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(_clean)
    return df

# ---------- 主函数 ----------
def build_clean(project_root: Path, *, force: bool = False):
    root   = Path(project_root)
    finals = root / "data" / "finals"
    cache  = root / "cache"
    cache.mkdir(exist_ok=True)

    # ① 读取
    f_nodes = finals / "collected_ext_nodes.csv"
    f_rels  = finals / "collected_ext_relations.csv"
    f_ents  = finals / "all_annotated_entities.csv"
    f_rls   = finals / "all_annotated_relationships.csv"

    df_nodes = pd.read_csv(f_nodes, low_memory=False)
    df_rels  = pd.read_csv(f_rels,  low_memory=False)
    df_ents  = pd.read_csv(f_ents,  low_memory=False)
    df_rls   = pd.read_csv(f_rls,   low_memory=False)

    log.info("✔ 原始读取完成")
    log.info(f"• collected_ext_nodes.csv  ({len(df_nodes):,} 行)")
    log.info(f"• collected_ext_relations.csv  ({len(df_rels):,} 行)")
    log.info(f"• all_annotated_entities.csv  ({len(df_ents):,} 行)")
    log.info(f"• all_annotated_relationships.csv  ({len(df_rls):,} 行)")

    # ② 清洗
    _clean_df(df_nodes, ["entity_name"])
    _clean_df(df_rels,  ["source_name", "target_name"])
    _clean_df(df_ents,  ["main_text"])
    _clean_df(df_rls,   ["source_main_text", "target_main_text"])

    # ③ 写 parquet（若已存在且非 force → 跳过）
    def _dump(df, out_name):
        out_path = cache / out_name
        if out_path.exists() and not force:
            log.warning(f"跳过 {out_name}（已存在，--force 可覆盖）")
            return
        df.to_parquet(out_path, index=False)
        log.info(f"📦 写出 {out_name}  ({len(df):,} rows)")

    _dump(df_nodes, "collected_ext_nodes_clean.parquet")
    _dump(df_rels,  "collected_ext_rels_clean.parquet")
    _dump(df_ents,  "annotated_entities_clean.parquet")
    _dump(df_rls,   "annotated_relationships_clean.parquet")

    rich.print("[bold green]🎉 清洗完毕[/]")

# ---------- Typer CLI ----------
app = typer.Typer()

@app.command()
def run(root: str = typer.Option(..., help="项目根目录"),
        force: bool = typer.Option(False, "--force/-f", help="已存在时覆盖")):
    """清洗并导出 Parquet"""
    build_clean(Path(root), force=force)

if __name__ == "__main__":
    app()
