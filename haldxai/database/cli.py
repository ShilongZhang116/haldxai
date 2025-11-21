"""
Command-line entry for wiping / importing databases.

> python -m haldxai.database.cli reset       # 清空两库
> python -m haldxai.database.cli reset --pg   # 只清 PG
> python -m haldxai.database.cli reset --neo  # 只清 Neo4j
> python -m haldxai.database.cli load        # 导入 data/database/*
"""
from __future__ import annotations
import typer
from pathlib import Path

from .pg_utils import wipe as wipe_pg, import_folder as import_pg
from .neo4j_utils import wipe as wipe_neo, import_nodes, import_relations

ROOT = Path(__file__).resolve().parents[2]          # project root
DATA_DIR = ROOT / "data" / "database"

app = typer.Typer(add_completion=False)

# ---------------- reset ----------------
@app.command()
def reset(pg: bool = True,
          neo: bool = True,
          drop_schema: bool = typer.Option(
              False, help="PG: drop & recreate schema")):
    """清空 PG / Neo4j"""
    if pg:
        wipe_pg(drop_schema=drop_schema)
    if neo:
        wipe_neo()

# ---------------- load -----------------
@app.command()
def load(nodes_csv: Path = DATA_DIR / "nodes.csv",
         rels_csv:  Path = DATA_DIR / "relations.csv"):
    """导入 data/database/* 到数据库"""
    # 1. PG 其它表
    import_pg(DATA_DIR, exclude=("nodes.csv", "relations.csv"))

    # 2. Neo4j
    import_nodes(nodes_csv)
    import_relations(rels_csv)

if __name__ == "__main__":
    app()
